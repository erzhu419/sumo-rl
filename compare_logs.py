import re

def parse_log(filepath):
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            # Split timestamp and content
            parts = line.split(' - ', 1)
            if len(parts) < 2:
                continue
            
            content = parts[1].strip()
            # Content format: TYPE       [Time:X] Details...
            # Split by whitespace to get type
            content_parts = content.split(None, 1)
            if len(content_parts) < 2:
                continue
            
            event_type = content_parts[0]
            details = content_parts[1]
            
            # Extract Bus ID
            # Bus:0_0 or Bus:0
            bus_match = re.search(r'Bus:([^_ ]+)', details)
            bus_id = bus_match.group(1) if bus_match else None
            
            # Extract Action if present
            action_match = re.search(r'Action:([\d\.-]+)', details)
            action_val = action_match.group(1) if action_match else None

            # Extract Reward if present
            reward_match = re.search(r'Reward:([\d\.-]+)', details)
            reward_val = reward_match.group(1) if reward_match else None

            # Extract Neighbor Info (ENV_DEBUG)
            # FwdBus:X BwdBus:Y or FwdTrip:X(ID) BwdTrip:Y(ID)
            fwd_match = re.search(r'Fwd(?:Bus|Trip):([^\s]+)', details)
            bwd_match = re.search(r'Bwd(?:Bus|Trip):([^\s]+)', details)
            
            # Extract Timings (ENV_TIMING / ENV_DEBUG)
            arrive_match = re.search(r'Arrive:([\d\.]+)', details)
            svc_end_match = re.search(r'(?:SvcEnd|ExchangeEnd):([\d\.]+)', details)
            obs_match = re.search(r'(?:StateObs|StateObsTime):([\d\.]+)', details)

            events.append({
                'type': event_type, 
                'details': details, 
                'bus_id': bus_id,
                'action': action_val,
                'reward': reward_val,
                'fwd': fwd_match.group(1) if fwd_match else None,
                'bwd': bwd_match.group(1) if bwd_match else None,
                'arrive': arrive_match.group(1) if arrive_match else None,
                'svc_end': svc_end_match.group(1) if svc_end_match else None,
                'obs': obs_match.group(1) if obs_match else None
            })
    return events

def analyze_sequence(events, bus_id_filter='0'):
    print(f"--- Event Sequence for Bus {bus_id_filter} ---")
    headers = f"{'TYPE':<15} | {'Arrive':<8} | {'SvcEnd':<8} | {'Obs':<8} | {'FwdBus':<12} | {'BwdBus':<12} | {'Action/Details'}"
    print(headers)
    print("-" * len(headers))
    
    count = 0
    for e in events:
        if e['bus_id'] == bus_id_filter:
            extra = ""
            if e['action']: extra += f"Act:{e['action']} "
            if e['reward']: extra += f"Rew:{e['reward']} "
            
            # For DEBUG/TIMING events, show raw details slightly truncated if too long
            if 'ENV_' in e['type']:
                 pass # Use columns
            
            row = f"{e['type']:<15} | {e['arrive'] or '':<8} | {e['svc_end'] or '':<8} | {e['obs'] or '':<8} | {e['fwd'] or '':<12} | {e['bwd'] or '':<12} | {extra}"
            print(row)
            
            count += 1
            if count > 40:
                print("... (truncated)")
                break

print("=== LEGACY LOG ANALYSIS (Bus 0) ===")
legacy_events = parse_log('legacy_debug.log')
analyze_sequence(legacy_events, bus_id_filter='0')

print("\n=== SUMO LOG ANALYSIS (Bus 7S) ===")
# SUMO logs ended up in debug_verification.log due to path/module issues, but they are complete.
sumo_events = parse_log('debug_verification.log')
analyze_sequence(sumo_events, bus_id_filter='7S_1') # Filter for specific bus seen in logs
