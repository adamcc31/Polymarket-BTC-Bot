import sys
import csv
import re

def parse_logs(input_path, output_path):
    markets = {} # market_id -> data

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract timestamp
                # Example: 2026-04-12T11:50:00.000Z
                ts_match = re.search(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z|(?:\+|-)\d{2}:\d{2})', line)
                if not ts_match:
                    ts_match = re.search(r'^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[^\s]*)', line)
                    
                timestamp = ts_match.group(1) if ts_match else None
                
                # Extract event name (first word after the bracketed log level)
                # Example: [info ] dynamic_5m_candidate_found
                event_match = re.search(r'\[.*?\]\s+([a-zA-Z0-9_]+)', line)
                if not event_match:
                    continue
                event = event_match.group(1)
                
                # Extract all key=value pairs
                # This regex captures keys and values, unquoting string values if necessary
                # We'll just grab non-space chunks or quoted strings
                kv_matches = re.finditer(r'([a-zA-Z0-9_]+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s]+))', line)
                kv_pairs = {}
                for m in kv_matches:
                    key = m.group(1)
                    val = m.group(2) or m.group(3) or m.group(4)
                    kv_pairs[key] = val
                
                market_id = kv_pairs.get('market_id')
                if not market_id:
                    continue
                    
                if market_id not in markets:
                    markets[market_id] = {
                        'timestamp_found': None,
                        'slug': None,
                        'market_id': market_id,
                        'strike_price': None,
                        'ttr_minutes_found': None,
                        'market_yes_prob': None,
                        'status': 'EVALUATING', # Default status if we only see random logs for this mid
                        'abstain_reason': None,
                        'live_edge': None,
                        'execution_price': None
                    }
                    
                market = markets[market_id]
                
                if event == 'dynamic_5m_candidate_found':
                    market['status'] = 'EVALUATING'
                    market['timestamp_found'] = timestamp
                    market['slug'] = kv_pairs.get('slug')
                    market['strike_price'] = kv_pairs.get('strike_price')
                    market['ttr_minutes_found'] = kv_pairs.get('TTR_minutes')
                    market['market_yes_prob'] = kv_pairs.get('yes_prob')
                    
                elif event == 'trade_aborted':
                    # Don't overturn EXECUTED since an abort might happen on a retry or parallel branch incorrectly
                    if market['status'] != 'EXECUTED':
                        market['status'] = 'ABSTAINED'
                        market['abstain_reason'] = kv_pairs.get('reason')
                        market['live_edge'] = kv_pairs.get('live_edge') or kv_pairs.get('edge')
                        
                elif event in ('trade_approved', 'paper_trade_opened', 'trade_executed'):
                    market['status'] = 'EXECUTED'
                    market['execution_price'] = kv_pairs.get('entry_price') or kv_pairs.get('bet_size')

    except FileNotFoundError:
        print(f"Error: Input log file '{input_path}' not found.")
        sys.exit(1)

    fields = [
        'timestamp_found', 'slug', 'market_id', 'strike_price', 
        'ttr_minutes_found', 'market_yes_prob', 'status', 
        'abstain_reason', 'live_edge', 'execution_price'
    ]
    
    exported_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for mid, data in markets.items():
            if data['slug']:  # Only export if Trigger 1 completed
                writer.writerow(data)
                exported_count += 1
                
    print(f"Successfully processed shadow book.")
    print(f"Total unique markets exported: {exported_count}")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scripts/export_shadow_book.py <input_log_path> <output_csv_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    parse_logs(input_path, output_path)
