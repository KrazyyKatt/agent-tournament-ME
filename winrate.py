import pandas as pd

SEP = "----"

def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def main(csv_path: str,
         winner_col: str = "winner",
         reason_col: str = "reason") -> None:

    df = pd.read_csv(csv_path)

    for c in (winner_col, reason_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    # normalize
    df["_winner_raw"] = df[winner_col].map(norm)
    df["_reason"] = df[reason_col].map(norm)

    # assign block index based on -------
    block = -1
    blocks = []

    for w in df["_winner_raw"]:
        if w == SEP:
            block += 1
            blocks.append(None)  # separator row
        else:
            if block == -1:
                block = 0
            blocks.append(block)

    df["_block"] = blocks

    # izbaci separator retke
    df = df[df["_block"].notna()].copy()
    df["_block"] = df["_block"].astype(int)

    # expected winner by block parity
    df["_expected"] = df["_block"].apply(lambda b: "blue" if b % 2 == 0 else "red")

    # ukupni zbroj razloga
    reason_counts = df["_reason"].value_counts()
    timeout = int(reason_counts.get("timeout", 0))
    flag_capture = int(reason_counts.get("flag_capture", 0))

    # winrate (samo decided)
    decided = df[df["_winner_raw"].isin(["blue", "red"])].copy()
    decided["_correct"] = decided["_winner_raw"] == decided["_expected"]

    decided_total = len(decided)
    correct = int(decided["_correct"].sum())
    wrong = decided_total - correct
    winrate = (correct / decided_total) * 100 if decided_total else 0.0

    # report
    print("=== TOTALS ===")
    print(f"Valid rows:   {len(df)}")
    print(f"flag_capture: {flag_capture}")
    print(f"timeout:      {timeout}")

    print("\n=== BLOCK-BASED WINRATE ===")
    print(f"Decided games: {decided_total}")
    print(f"Correct:       {correct}")
    print(f"Wrong:         {wrong}")
    print(f"Winrate:       {winrate:.2f}%")

    if wrong > 0:
        print("\n=== MISMATCHES ===")
        print(
            decided.loc[~decided["_correct"],
                        ["_block", winner_col, reason_col, "_expected"]]
            .rename(columns={"_block": "block", "_expected": "expected"})
            .to_string(index=False)
        )

if __name__ == "__main__":
    main("results.csv")
