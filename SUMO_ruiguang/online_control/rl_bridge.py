import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import time
import random
import argparse
import math
import numpy as np
from typing import Tuple, List, Dict, Set, Any, Callable, Optional

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")

try:
    import libsumo
    sys.modules['traci'] = libsumo
    import traci
    print("Using libsumo for simulation (globally patched).")
    LIBSUMO = True
except ImportError:
    import traci
    print("Using traci for simulation.")
    LIBSUMO = False

from sumolib import checkBinary
import traci.constants as tc

from .case import f_8_create_obj
from .case import e_8_gurobi_test_considerbusnum_V3
from .case import d_8_compute_running_time

from .rl_env import DecisionEvent


class SumoRLBridge:
    def __init__(
        self,
        root_dir: str,
        sumo_cfg: str = "control_sim_traci_period.sumocfg",
        *,
        gui: bool = False,
        max_steps: int = 18000,
        time_step: float = 1.0,
        default_headway: float = 360.0,
        update_freq: int = 10,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.sumo_cfg = sumo_cfg if os.path.isabs(sumo_cfg) else os.path.join(self.root_dir, sumo_cfg)
        self.gui = gui
        self.max_steps = max_steps
        self.time_step = time_step
        self.default_headway = default_headway
        self.update_freq = update_freq

        self.initialized = False
        self.first_run = True
        self.done = False
        self.current_time = 0.0
        self.steps = 0

        self.decision_queue: deque[DecisionEvent] = deque()
        self.pending_events: Dict[Tuple[str, str, str], DecisionEvent] = {}
        self.active_events: Dict[str, DecisionEvent] = {}

        self.arrival_history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.depart_history: Dict[str, Dict[str, float]] = defaultdict(dict)

        self.stop_indices: Dict[str, Dict[str, int]] = defaultdict(dict)

        self.sumo_binary = None

        self.lane_obj_dic = {}
        self.stop_obj_dic = {}
        self.signal_obj_dic = {}
        self.line_obj_dic = {}
        self.bus_obj_dic = {}
        self.passenger_obj_dic = {}

        self.BusCap = 50
        self.AveAlightingTime = 1.5
        self.AveBoardingTime = 2.5

        self.busline_edge_order = {}
        self.involved_tl_ID_l = []
        self.line_headways = {}  # Store median headway per line

    # region SUMO lifecycle
    def reset(self) -> None:
        # Optimization: Load network once and reuse state
        if self.first_run:
            if self.initialized:
                 self.close()
            self._ensure_sumo_home()
            self._start_traci()
            self._load_objects()
            # Save state for future resets
            if LIBSUMO:
                traci.simulation.saveState("sumo_start_state.sbx")
            self.first_run = False
        else:
            # Soft reset using saved state
            if LIBSUMO:
                # Suppress "Loading state..." output
                with open(os.devnull, 'w') as devnull:
                    try:
                        # Redirect stdout (fd 1) and stderr (fd 2) at the OS level
                        # This is necessary because libsumo prints from C++
                        old_stdout = os.dup(1)
                        old_stderr = os.dup(2)
                        os.dup2(devnull.fileno(), 1)
                        os.dup2(devnull.fileno(), 2)
                        try:
                            traci.simulation.loadState("sumo_start_state.sbx")
                        finally:
                            os.dup2(old_stdout, 1)
                            os.dup2(old_stderr, 2)
                            os.close(old_stdout)
                            os.close(old_stderr)
                    except Exception:
                        # Fallback if redirection fails
                        traci.simulation.loadState("sumo_start_state.sbx")
            else:
                # Fallback for standard traci (if needed, though we focus on libsumo)
                self.close()
                self._ensure_sumo_home()
                self._start_traci()
            
            # Always reload objects to ensure fresh python state matching the simulation
            self._load_objects()

        self.initialized = True
        self.done = False
        self.current_time = traci.simulation.getTime()
        self.steps = 0
        self.decision_queue.clear()
        self.pending_events.clear()
        self.active_events.clear()
        self.arrival_history.clear()
        self.depart_history.clear()
        
        # Optimization: Track active agents locally to avoid expensive getIDList() calls
        self.active_bus_ids = set()
        self.active_passenger_ids = set()
        self.just_departed_buses = []

    def close(self) -> None:
        # Optimization: Do NOT close traci if using libsumo reuse strategy
        if not LIBSUMO:
            if traci.isLoaded():
                traci.close()
        
        self.initialized = False
        self.active_bus_ids = set()
        self.active_passenger_ids = set()

    def _ensure_sumo_home(self) -> None:
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

    def _start_traci(self) -> None:
        # Optimization: If traci is already loaded, don't start it again
        if LIBSUMO and traci.isLoaded():
            return

        if LIBSUMO:
            # libsumo runs in-process, no binary needed, but argv[0] is expected
            # Added --duration-log.disable and --log /dev/null to suppress verbose output
            traci.start(["sumo", "-c", self.sumo_cfg, "--no-warnings", "--duration-log.disable", "--log", "/dev/null"])
            self.sumo_binary = "libsumo"
        else:
            binary = checkBinary('sumo-gui' if self.gui else 'sumo')
            traci.start([binary, "-c", self.sumo_cfg, "--no-warnings", "--duration-log.disable", "--log", "/dev/null"])
            self.sumo_binary = binary

    def _load_objects(self) -> None:
        result = f_8_create_obj.create_obj_fun()
        self.lane_obj_dic = result[0]
        self.stop_obj_dic = result[1]
        self.signal_obj_dic = result[2]
        self.line_obj_dic = result[3]
        self.bus_obj_dic = result[4]
        self.passenger_obj_dic = result[5]
        for bus in self.bus_obj_dic.values():
            bus.last_forward_headway = None  # Store the last calculated forward headway (for backward bus logic)
            if not hasattr(bus, 'trajectory_dict'):
                bus.trajectory_dict = defaultdict(list)

        (
            _,
            _,
            _,
            _,
            line_station_od_otd_dict,
            bus_arrstation_od_otd_dict,
            _,
            self.BusCap,
            self.AveAlightingTime,
            self.AveBoardingTime,
        ) = e_8_gurobi_test_considerbusnum_V3.get_static_info(self.line_obj_dic)
        self.bus_arrstation_od_otd_dict = bus_arrstation_od_otd_dict

        edge_file_path = os.path.join(self.root_dir, "intersection_delay", "a_sorted_busline_edge.xml")
        edge_file = ET.parse(edge_file_path)
        sorted_busline_edge_d, involved_tl_ID_l, _, _ = d_8_compute_running_time.get_sorted_busline_edge(edge_file)
        self.busline_edge_order = sorted_busline_edge_d
        self.involved_tl_ID_l = involved_tl_ID_l

        trigger_time = 900
        for stop_id, stop_obj in self.stop_obj_dic.items():
            stop_obj.get_accessible_stop(self.line_obj_dic)
            stop_obj.get_passenger_arriver_rate(self.passenger_obj_dic)
            stop_obj.get_initial_just_leave_data(line_station_od_otd_dict, trigger_time)
        
        # Optimization: Skip signal object initialization as it is unused for Holding Control
        for signal_id, signal_obj in self.signal_obj_dic.items():
            signal_obj.get_attribute_by_traci()
            signal_obj.get_pass_line(self.line_obj_dic)
            
        for bus_id, bus_obj in self.bus_obj_dic.items():
            line = self.line_obj_dic[bus_obj.belong_line_id_s]
            bus_obj.get_arriver_timetable(line)

        for line_id, line in self.line_obj_dic.items():
            self.stop_indices[line_id] = {stop_id: idx for idx, stop_id in enumerate(line.stop_id_l)}

        # Compute median headway for each line
        for line_id, line in self.line_obj_dic.items():
            # Find all buses for this line
            buses = [b for b in self.bus_obj_dic.values() if b.belong_line_id_s == line_id]
            if len(buses) > 1:
                buses.sort(key=lambda b: b.start_time_n)
                start_times = [b.start_time_n for b in buses]
                diffs = [j - i for i, j in zip(start_times[:-1], start_times[1:])]
                if diffs:
                    self.line_headways[line_id] = float(np.median(diffs))
                else:
                    self.line_headways[line_id] = self.default_headway
            else:
                self.line_headways[line_id] = self.default_headway
        # print(f"Computed Line Headways: {self.line_headways}")

    # endregion

    # region RL hooks
    def fetch_events(self) -> Tuple[List[DecisionEvent], bool, List[str]]:
        if self.done:
            return [], True, []
        if not self.initialized:
            self.reset()
        if not self.decision_queue:
            self._advance_until_event()
        if not self.decision_queue and self.done:
            return [], True, []
        events = list(self.decision_queue)
        self.decision_queue.clear()
        
        departed = list(self.just_departed_buses)
        self.just_departed_buses.clear()
        
        return events, self.done, departed

    def apply_action(self, event: DecisionEvent, hold_value: float) -> None:
        if not self.initialized or self.done:
            return
        duration = max(event.base_stop_duration + float(hold_value), 0.0)
        bus_id = event.bus_id
        stop_id = event.stop_id
        stopping_place = event.metadata.get('stopping_place', stop_id)
        
        try:
            # Since we applied pre-emptive holding (duration=3600) in _collect_new_events,
            # the stop should still be active and at index 0.
            # We simply update it to the actual desired duration.
            traci.vehicle.setStopParameter(bus_id, 0, "duration", str(duration))
        except traci.TraCIException as e:
            # If it still fails, log it but don't crash.
            # This might happen if the bus somehow left the stop despite our best efforts.
            print(f"WARNING: Failed to apply holding to {bus_id} at {stop_id}. Error: {e}")
            pass

        key = (event.line_id, bus_id, stop_id)
        if key not in self.pending_events:
            self.pending_events[key] = event
        self.active_events[bus_id] = event
        event.assigned_action = hold_value
        bus_obj = self.bus_obj_dic.get(bus_id)
        if bus_obj and stop_id in bus_obj.just_server_stop_data_d:
            arrive_time = bus_obj.just_server_stop_data_d[stop_id][0]
            bus_obj.just_server_stop_data_d[stop_id][1] = arrive_time + duration
        event.metadata['applied_duration'] = duration

    # endregion

    # region simulation stepping
    def _advance_until_event(self) -> None:
        while not self.done and not self.decision_queue:
            self._advance_one_step()

    def _advance_one_step(self) -> None:
        if self.done:
            return
        simulation_current_time = traci.simulation.getTime()
        
        # Update active buses
        if self.steps == 0:
            # On the first step, check ALL existing vehicles to catch those that departed at time 0
            initial_ids = traci.vehicle.getIDList()
            for veh_id in initial_ids:
                if veh_id in self.bus_obj_dic:
                    self.active_bus_ids.add(veh_id)
    
        
        departed_ids = traci.simulation.getDepartedIDList()
        for veh_id in departed_ids:
            if veh_id in self.bus_obj_dic:
                self.active_bus_ids.add(veh_id)

        
        arrived_ids = traci.simulation.getArrivedIDList()
        for veh_id in arrived_ids:
            if veh_id in self.active_bus_ids:
                self.active_bus_ids.discard(veh_id)
                self.just_departed_buses.append(veh_id)
                
        # Update active passengers
        departed_persons = traci.simulation.getDepartedPersonIDList()
        for person_id in departed_persons:
            if person_id in self.passenger_obj_dic:
                self.active_passenger_ids.add(person_id)
                
        arrived_persons = traci.simulation.getArrivedPersonIDList()
        for person_id in arrived_persons:
            if person_id in self.active_passenger_ids:
                self.active_passenger_ids.discard(person_id)

        # Optimization: Update stops and passengers only every update_freq steps
        if self.steps % self.update_freq == 0:
            for stop in self.stop_obj_dic.values():
                stop.update_stop_state()
            
            # Iterate only active passengers
            for passenger_id in list(self.active_passenger_ids):
                passenger_obj = self.passenger_obj_dic[passenger_id]
                if passenger_obj.passenger_state_s == "No":
                    passenger_obj.passenger_activate(simulation_current_time, self.line_obj_dic)
                else:
                    passenger_obj.passenger_run(simulation_current_time, self.line_obj_dic)

        # CRITICAL: Bus logic must run EVERY step to detect arrivals/departures correctly
        # Otherwise, short stops (duration=0) are missed if they fall between update intervals
        for vehicle_id in list(self.active_bus_ids):
            bus_obj = self.bus_obj_dic[vehicle_id]
            line_obj = self.line_obj_dic[bus_obj.belong_line_id_s]
            if bus_obj.bus_state_s == "No":
                bus_obj.bus_activate(line_obj, self.stop_obj_dic, self.signal_obj_dic, simulation_current_time)
            else:
                bus_obj.bus_running(
                    line_obj,
                    self.stop_obj_dic,
                    self.signal_obj_dic,
                    self.passenger_obj_dic,
                    simulation_current_time,
                    self.BusCap,
                    self.AveAlightingTime,
                    self.AveBoardingTime,
                    self.bus_arrstation_od_otd_dict,
                    self.bus_obj_dic,
                    self.involved_tl_ID_l,
                    self.busline_edge_order,
                )

        self._collect_new_events(simulation_current_time)

        traci.simulationStep()
        self.current_time = traci.simulation.getTime()
        self.steps += 1

        self._cleanup_departed_buses()
        self._check_termination()

    def _find_neighbors(self, line_id: str, current_bus_id: str) -> Tuple[Optional[Any], Optional[Any]]:
        # Find all buses on this line
        buses = [b for b in self.bus_obj_dic.values() if b.belong_line_id_s == line_id]
        # Sort by start_time_n (earlier start time = forward bus)
        buses.sort(key=lambda b: b.start_time_n)
        
        try:
            current_idx = [b.bus_id_s for b in buses].index(current_bus_id)
        except ValueError:
            return None, None
            
        forward_bus = buses[current_idx - 1] if current_idx > 0 else None
        backward_bus = buses[current_idx + 1] if current_idx < len(buses) - 1 else None
        
        return forward_bus, backward_bus

    def _compute_robust_headway(self, subject_bus, front_bus, stop_id, current_time, target_headway=360.0, force_distance=False) -> float:
        if not front_bus or not subject_bus:
            return target_headway
            
        # 1. Try trajectory-based calculation (if front bus has arrived at this stop)
        if not force_distance and stop_id in front_bus.trajectory_dict and front_bus.trajectory_dict[stop_id]:
            front_arrive_time = front_bus.trajectory_dict[stop_id][-1]
            return current_time - front_arrive_time
            
        # 2. Fallback to distance-based calculation
        dist_diff = subject_bus.distance_n - front_bus.distance_n
        
        # Calculate subject average speed
        duration = current_time - subject_bus.start_time_n
        if duration <= 1.0:
             return target_headway
             
        avg_speed = subject_bus.distance_n / duration
        if avg_speed < 0.1:
            return target_headway
            
        headway = -dist_diff / avg_speed
        return max(headway, 0.0) # Ensure positive

    def _collect_new_events(self, simulation_current_time: float) -> None:
        for bus_id, bus_obj in self.bus_obj_dic.items():
            if bus_obj.bus_state_s != "Stop" or not bus_obj.just_server_stop_data_d:
                continue
            stop_id = next(iter(bus_obj.just_server_stop_data_d.keys()))
            key = (bus_obj.belong_line_id_s, bus_id, stop_id)
            if key in self.pending_events:
                continue
            arrive_time, predicted_depart_time = bus_obj.just_server_stop_data_d[stop_id]
            base_duration = max(predicted_depart_time - arrive_time, 0.0)
            
            # Timing Alignment: Only process if passenger exchange is done
            # This ensures we observe the state AFTER passengers have boarded/alighted, matching original logic
            if simulation_current_time < arrive_time + base_duration:
                continue

            # Prevent duplicate events for the same stop visit
            if getattr(bus_obj, 'last_action_time', -1.0) == arrive_time:
                continue
            bus_obj.last_action_time = arrive_time

            line_id = bus_obj.belong_line_id_s
            
            # Find neighbors for robust headway calculation
            forward_bus, backward_bus = self._find_neighbors(line_id, bus_id)
            
            # "No Control" Logic: Skip isolated buses
            if not forward_bus and not backward_bus:
                continue
                
            # Dynamic Target Headway Calculation
            default_line_headway = self.line_headways.get(line_id, self.default_headway)
            
            target_forward = default_line_headway
            if forward_bus:
                target_forward = abs(bus_obj.start_time_n - forward_bus.start_time_n)
                
            target_backward = default_line_headway
            if backward_bus:
                target_backward = abs(backward_bus.start_time_n - bus_obj.start_time_n)
                
            # Service Completion Time: When passenger exchange finishes (Arrive + Duration)
            # This is the exact time point used for Headway calculation in LSTM-RL
            service_completion_time = arrive_time + base_duration

            # Robust Headway Calculation
            # Forward Headway: Me - Front
            # Use service_completion_time instead of simulation_current_time for precision
            forward_headway = self._compute_robust_headway(bus_obj, forward_bus, stop_id, service_completion_time, target_headway=target_forward)
            
            # Store this forward headway for the backward bus to use later
            bus_obj.last_forward_headway = forward_headway

            # Backward Headway: Back - Me
            # Note: We pass backward_bus as subject, bus_obj as front
            # CRITICAL UPDATE: Use the backward bus's *stored* forward headway (stale but exact), matching LSTM-RL logic.
            # If backward bus hasn't recorded a headway yet (e.g. just started), fallback to target.
            if backward_bus and backward_bus.last_forward_headway is not None:
                backward_headway = backward_bus.last_forward_headway
            else:
                backward_headway = target_backward

            stop_idx = self.stop_indices[line_id].get(stop_id, -1)
            direction = 1 if line_id.endswith('S') else 0
            waiting_passengers = len(traci.busstop.getPersonIDs(stop_id))
            
            stop_data = traci.vehicle.getStops(bus_id, 1)
            stopping_place = stop_data[0].stoppingPlaceID if stop_data else stop_id
            
            event = DecisionEvent(
                line_id=line_id,
                bus_id=bus_id,
                stop_id=stop_id,
                stop_idx=stop_idx,
                direction=direction,
                sim_time=simulation_current_time,
                forward_headway=forward_headway,
                backward_headway=backward_headway,
                waiting_passengers=waiting_passengers,
                base_stop_duration=base_duration,
                forward_bus_present=bool(forward_bus),
                backward_bus_present=bool(backward_bus),
                target_forward_headway=target_forward,
                target_backward_headway=target_backward,
                metadata={'arrive_time': arrive_time, 'stopping_place': stopping_place},
            )
            
            # Pre-emptive Holding:
            # Immediately set a large duration to keep the stop active in SUMO while the agent decides.
            # This prevents the "not downstream" error caused by SUMO finishing the stop prematurely.
            try:
                # Use setStopParameter to modify the current stop (index 0)
                traci.vehicle.setStopParameter(bus_id, 0, "duration", "3600.0")
            except traci.TraCIException:
                pass

            self.decision_queue.append(event)
            self.pending_events[key] = event
            self.arrival_history[line_id][stop_id].append(arrive_time)
            
            # Populate trajectory_dict with SERVICE COMPLETION time
            # This matches LSTM-RL logic: current_time + holding_time
            bus_obj.trajectory_dict[stop_id].append(service_completion_time)

    def _cleanup_departed_buses(self) -> None:
        to_remove = []
        for bus_id, event in self.active_events.items():
            bus_obj = self.bus_obj_dic.get(bus_id)
            if not bus_obj or bus_obj.bus_state_s != "Stop":
                stop_id = event.stop_id
                line_id = event.line_id
                if bus_obj and stop_id in bus_obj.depart_stop_time_d:
                    depart_time = bus_obj.depart_stop_time_d[stop_id]
                    self.depart_history[line_id][stop_id] = depart_time
                key = (line_id, bus_id, stop_id)
                self.pending_events.pop(key, None)
                to_remove.append(bus_id)
        for bus_id in to_remove:
            self.active_events.pop(bus_id, None)

    def _check_termination(self) -> None:
        if self.steps >= self.max_steps:
            self.done = True
        elif traci.simulation.getMinExpectedNumber() <= 0:
            self.done = True
        if self.done:
            self._cleanup_departed_buses()
            self.close()

    def _compute_forward_headway(self, line_id: str, stop_id: str, arrive_time: float) -> float:
        history = self.arrival_history[line_id][stop_id]
        if history:
            last_time = history[-1]
            gap = arrive_time - last_time
            return gap if gap > 0 else self.default_headway
        return self.default_headway

    def _compute_backward_headway(self, line_id: str, stop_id: str) -> float:
        depart_time = self.depart_history[line_id].get(stop_id)
        arrive_times = self.arrival_history[line_id][stop_id]
        if depart_time is not None and arrive_times:
            gap = depart_time - arrive_times[-1]
            if gap > 0:
                return gap
        return self.default_headway

    # endregion







def build_bridge(*, root_dir: str, sumo_cfg: str = "control_sim_traci_period.sumocfg", gui: bool = False, update_freq: int = 10) -> Dict[str, Callable]:
    bridge = SumoRLBridge(root_dir=root_dir, sumo_cfg=sumo_cfg, gui=gui, update_freq=update_freq)

    def decision_provider() -> Tuple[List[DecisionEvent], bool]:
        return bridge.fetch_events()

    def action_executor(event: DecisionEvent, action_value: float) -> None:
        bridge.apply_action(event, action_value)

    def reset_callback() -> None:
        bridge.reset()

    def close_callback() -> None:
        bridge.close()

    return {
        'decision_provider': decision_provider,
        'action_executor': action_executor,
        'reset_callback': reset_callback,
        'close_callback': close_callback,
    }

