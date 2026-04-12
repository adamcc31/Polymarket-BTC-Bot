import csv
import itertools

def optimize():
    # Load dataset
    trades = []
    with open('../dataset/trade_12042026.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                'live_edge': float(row['live_edge']),
                'TTR_minutes': float(row['TTR_minutes']),
                'pnl_usd': float(row['pnl_usd']),
                'outcome': row['outcome']
            })

    # Grid parameters
    margins_of_safety = [0.02, 0.03, 0.04, 0.05]
    max_live_edges = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    min_ttr_minutes_list = [1.5, 2.0, 3.0, 3.5]
    
    results = []
    
    for margin, max_edge, min_ttr in itertools.product(margins_of_safety, max_live_edges, min_ttr_minutes_list):
        accepted_trades = 0
        wins = 0
        total_pnl = 0.0
        
        for t in trades:
            if t['live_edge'] < margin:
                continue
            if t['live_edge'] > max_edge:
                continue
            if t['TTR_minutes'] < min_ttr:
                continue
                
            accepted_trades += 1
            if t['outcome'] == 'WIN':
                wins += 1
            total_pnl += t['pnl_usd']
            
        win_rate = (wins / accepted_trades * 100.0) if accepted_trades > 0 else 0.0
        
        results.append({
            'margin_of_safety': margin,
            'max_live_edge': max_edge,
            'min_ttr_minutes': min_ttr,
            'trade_count': accepted_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl
        })
        
    # Sort by total_pnl descending
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    
    # Print Top 3
    print("=== TOP 3 CONFIGURATIONS ===")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"Rank {i+1}:")
        print(f"  margin_of_safety: {r['margin_of_safety']}")
        print(f"  max_live_edge   : {r['max_live_edge']}")
        print(f"  min_ttr_minutes : {r['min_ttr_minutes']}")
        print(f"  ---------------------")
        print(f"  Trade Count : {r['trade_count']}")
        print(f"  Win Rate    : {r['win_rate']:.2f}%")
        print(f"  Total PnL   : ${r['total_pnl']:.2f}")
        print()

if __name__ == '__main__':
    optimize()
