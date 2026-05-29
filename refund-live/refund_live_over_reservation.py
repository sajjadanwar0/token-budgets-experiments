import csv, glob, os, statistics, sys

def main(argv):
    d = argv[1] if len(argv) > 1 else '.'
    files = sorted(glob.glob(os.path.join(d, '*.csv')))
    if not files:
        print("No CSVs found in", d); return 1
    total_rows = 0
    ratios = []
    with_ratio = 0
    for f in files:
        rows = list(csv.DictReader(open(f, newline='')))
        total_rows += len(rows)
        fn = (csv.DictReader(open(f, newline='')).fieldnames or [])
        if 'margin_ratio' not in fn:
            continue
        with_ratio += len(rows)
        for r in rows:
            try:
                x = float(r.get('margin_ratio', ''))
            except (ValueError, TypeError):
                continue
            if x > 0:
                ratios.append(x)
    print(f"Total live-API row-event corpus:      {total_rows}")
    print(f"Per-call over-reservation sample (N): {len(ratios)}")
    print(f"  mean   over-reservation: {statistics.mean(ratios):.2f}x")
    print(f"  median over-reservation: {statistics.median(ratios):.2f}x")
    print(f"  range:                   {min(ratios):.2f}x - {max(ratios):.2f}x")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except BrokenPipeError:
        pass