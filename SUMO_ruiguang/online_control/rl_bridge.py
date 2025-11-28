import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Callable, Dict, List, Tuple, Optional

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")

import traci
from sumolib import checkBinary

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
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.sumo_cfg = sumo_cfg if os.path.isabs(sumo_cfg) else os.path.join(self.root_dir, sumo_cfg)
        self.gui = gui
        self.max_steps = max_steps
        self.time_step = time_step
        self.default_headway = default_headway

        self.initialized = False
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

    # region SUMO lifecycle
    def reset(self) -> None:
        if self.initialized:
            self.close()
        self._ensure_sumo_home()
        self._start_traci()
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

    def close(self) -> None:
        if traci.isLoaded():
            traci.close()
        self.initialized = False

    def _ensure_sumo_home(self) -> None:
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'!")
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

    def _start_traci(self) -> None:
        binary = checkBinary('sumo-gui' if self.gui else 'sumo')
        traci.start([binary, "-c", self.sumo_cfg])
        self.sumo_binary = binary

    def _load_objects(self) -> None:
        result = f_8_create_obj.create_obj_fun()
        self.lane_obj_dic = result[0]
        self.stop_obj_dic = result[1]
        self.signal_obj_dic = result[2]
        self.line_obj_dic = result[3]
        self.bus_obj_dic = result[4]
        self.passenger_obj_dic = result[5]

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
        for signal_id, signal_obj in self.signal_obj_dic.items():
            signal_obj.get_attribute_by_traci()
            signal_obj.get_pass_line(self.line_obj_dic)
        for bus_id, bus_obj in self.bus_obj_dic.items():
            line = self.line_obj_dic[bus_obj.belong_line_id_s]
            bus_obj.get_arriver_timetable(line)

        for line_id, line in self.line_obj_dic.items():
            self.stop_indices[line_id] = {stop_id: idx for idx, stop_id in enumerate(line.stop_id_l)}

    # endregion

    # region RL hooks
    def fetch_events(self) -> Tuple[List[DecisionEvent], bool]:
        if self.done:
            return [], True
        if not self.initialized:
            self.reset()
        if not self.decision_queue:
            self._advance_until_event()
        if not self.decision_queue and self.done:
            return [], True
        events = list(self.decision_queue)
        self.decision_queue.clear()
        return events, self.done

    def apply_action(self, event: DecisionEvent, hold_value: float) -> None:
        if not self.initialized or self.done:
            return
        duration = max(event.base_stop_duration + float(hold_value), 0.0)
        bus_id = event.bus_id
        stop_id = event.stop_id
        stopping_place = event.metadata.get('stopping_place', stop_id)
        try:
            traci.vehicle.setStopParameter(bus_id, 0, "duration", str(duration))
        except traci.TraCIException:
            try:
                traci.vehicle.setBusStop(bus_id, stopping_place, duration=duration)
            except traci.TraCIException:
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
        for stop in self.stop_obj_dic.values():
            stop.update_stop_state()
        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            if traci.vehicle.getTypeID(vehicle_id) != "Bus":
                continue
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
        passenger_ids = traci.person.getIDList()
        for passenger_id in passenger_ids:
            passenger_obj = self.passenger_obj_dic[passenger_id]
            if passenger_obj.passenger_state_s == "No":
                passenger_obj.passenger_activate(simulation_current_time, self.line_obj_dic)
            else:
                passenger_obj.passenger_run(simulation_current_time, self.line_obj_dic)

        self._collect_new_events(simulation_current_time)

        traci.simulationStep()
        self.current_time = traci.simulation.getTime()
        self.steps += 1

        self._cleanup_departed_buses()
        self._check_termination()

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
            line_id = bus_obj.belong_line_id_s
            stop_idx = self.stop_indices[line_id].get(stop_id, -1)
            direction = 1 if line_id.endswith('S') else 0
            waiting_passengers = len(traci.busstop.getPersonIDs(stop_id))
            forward_headway = self._compute_forward_headway(line_id, stop_id, arrive_time)
            backward_headway = self._compute_backward_headway(line_id, stop_id)
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
                metadata={'arrive_time': arrive_time, 'stopping_place': stopping_place},
            )
            self.decision_queue.append(event)
            self.pending_events[key] = event
            self.arrival_history[line_id][stop_id].append(arrive_time)

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


def build_bridge(*, root_dir: str, sumo_cfg: str = "control_sim_traci_period.sumocfg", gui: bool = False) -> Dict[str, Callable]:
    bridge = SumoRLBridge(root_dir=root_dir, sumo_cfg=sumo_cfg, gui=gui)

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
