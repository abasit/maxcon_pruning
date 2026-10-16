"""The paper's synthetic tables, from the records in results/ and results/sweep/ (read only), into results/tables/:

  grid_instances.csv, sweep_instances.csv   one row per instance with every number below, the layer the tables are
                      cut from
  tab_grid.tex        Table 1, Algorithm 1 on the grid per cell (d, t): survivors, inliers removed, outliers left and
                      time; A* on the survivors; A* alone, runs and time
  tab_trimming.tex    Table 2, the trimming at k eps per cell, k in trimming.KS: survivors, inliers removed, outliers
                      left and A* time on the trimmed set
  tab_wi_mbf.tex      Table 3, WI and MBF-MaxCon per cell: m*, then per method its returned set as survivors, inliers
                      removed and outliers left, and its time
  tab_sweep_a.tex, tab_sweep_b.tex   Table 4, the residual sweep, one row per instance (k up to 5, then from 10)
  tab_gore.tex        Table 5, GORE at its published settings per cell: runs, survivors, inliers removed, outliers
                      left, time and runs at the limit
  tables.md           the same tables in markdown

A cell's three numbers are min/median/max over its instances. Inliers removed are the points of the optimal set that a
method's output lacks, and outliers left the planted outliers it keeps. The optimal set is the set of a finished A*
run, on the full instance if that finished, else on the survivors (an optimum with probability at least 1 - DELTA);
m* is its size. Times are in seconds, LO-RANSAC's included for the methods that start from it (the pruning and GORE).
>3,500 is a run at the A* limit, with no answer, and a dash a number not available: a run not made yet, or no optimal
set known. The LaTeX needs booktabs and the paper's \\Astar macro.

    python -m maxcon_pruning.synthetic.tables
"""
import csv
import math
import statistics as st

from .. import astar, common, trimming
from .instances import DS, GRID, N, ROOT, SWEEP_NAMES, TS, load, parse

RES, OUT = ROOT / "results", ROOT / "results" / "tables"
METHODS = (("wi", "WI", "WI"), ("mbf", "MBF", "MBF-MaxCon"))       # CSV prefix, the records' method, the table's name


def a_time(a):
    """An A* line's time: None without a line or for one in error, inf at the limit."""
    if a is None or "error" in a:
        return None
    return math.inf if a["hit_limit"] else a["time"]


def optimal(runs, name):
    """The optimal set of an instance (see the module docstring), or None."""
    for arm in ("alone", "surv"):
        a = runs.get(astar.job(name, arm))
        if a is not None and "error" not in a and not a["hit_limit"]:
            return set(a["set"])
    return None


def instance_rows(names, res):
    """One dict per instance with every number the tables use; None where a record is missing."""
    lo = {r["name"]: r for r in common.read(res / "lo.jsonl")}
    pr = {r["name"]: r for r in common.read(res / "prune.jsonl")}
    tr = {(r["name"], r["k"]): r for r in common.read(res / "trimming.jsonl")}
    go = {r["name"]: r for r in common.read(res / "gore.jsonl")}
    inf = {}
    for r in common.read(res / "influence.jsonl"):
        if "error" not in r:
            inf.setdefault((r["name"], r["method"]), r)
    runs = astar.runs(res / "astar.jsonl")
    rows = []
    for nm in names:
        d, t, run, k = parse(nm)
        outlier, opt = load(nm)["outlier"], optimal(runs, nm)

        def lost(S):
            return None if opt is None else len(opt - set(S))

        row = dict(name=nm, d=d, t=t, run=run, k=k, m0=lo[nm]["m0"] if nm in lo else None,
                   m_star=None if opt is None else len(opt))
        p = pr.get(nm)
        row.update(prune_survivors=p and p["survivors"], prune_inliers_removed=p and lost(p["keep"]),
                   prune_outliers_left=p and p["outliers_left"], prune_lower=p and p["cert_lower"],
                   prune_upper=p and p["cert_upper"], prune_exact=p and p["exact"], prune_samples=p and p["samples"],
                   prune_seconds=p and p["seconds"], astar_surv_time=a_time(runs.get(astar.job(nm, "surv"))),
                   astar_alone_run=astar.job(nm, "alone") in runs,
                   astar_alone_time=a_time(runs.get(astar.job(nm, "alone"))))
        for kk in trimming.KS:
            x = tr.get((nm, kk))
            row.update({f"trim{kk}_survivors": x and x["kept"], f"trim{kk}_inliers_removed": x and lost(x["keep"]),
                        f"trim{kk}_outliers_left": x and x["outliers_kept"],
                        f"trim{kk}_astar_time": a_time(runs.get(astar.job(nm, f"k{kk}")))})
        g = go.get(nm)
        if g is not None:
            keep = sorted(set(range(g["n"])) - set(g["removed"]))
        row.update(gore_survivors=g and g["survivors"], gore_inliers_removed=g and lost(keep),
                   gore_outliers_left=g and g["outliers_left"], gore_seconds=g and g["seconds"],
                   gore_at_limit=g and g["at_limit"])
        for pre, method, _ in METHODS:
            w = inf.get((nm, method))
            row.update({f"{pre}_survivors": w and w["size"], f"{pre}_inliers_removed": w and lost(w["returned"]),
                        f"{pre}_outliers_left": w and int(outlier[w["returned"]].sum()),
                        f"{pre}_seconds": w and w["seconds"]})
        rows.append(row)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if v is None else v for k, v in r.items()})


def fmt(v, digits, tex):
    """A number as a table shows it: a dash for None, >3,500 for a run at the A* limit."""
    if v is None:
        return "--" if tex else "-"
    if isinstance(v, float) and math.isinf(v):
        return "$>$3,500" if tex else ">3,500"
    if isinstance(v, bool):
        return str(int(v))
    return f"{v:,}" if isinstance(v, int) else f"{v:,.{digits}f}"


def mmm(values, digits, tex):
    """min/median/max of the values present, a dash when none is."""
    v = [x for x in values if x is not None]
    if not v:
        return fmt(None, digits, tex)
    return "/".join(fmt(x, digits, tex) for x in (min(v), st.median(v), max(v)))


def two(a, b):
    """A two-line column header."""
    return f"\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{a}\\\\{b}\\end{{tabular}}"


IR, OL = two("inliers", "removed"), two("outliers", "left")


def tabular(name, colspec, head, rows, top=False):
    """Write results/tables/<name>.tex, one whole tabular."""
    lines = [f"\\begin{{tabular}}{'[t]' if top else ''}{{{colspec}}}", "\\toprule", *head, "\\midrule", *rows,
             "\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


class Markdown:
    """The markdown file, one section per table."""

    def __init__(self, title):
        self.lines = [f"# {title}\n", "From the records by `python -m maxcon_pruning.synthetic.tables`. Cells a/b/c "
                      "are min/median/max over the instances of a cell. Times in seconds, >3,500 a run at the A* "
                      "limit, a dash a number not available.\n"]

    def table(self, title, header, rows):
        self.lines += [f"\n## {title}\n", "| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
        self.lines += ["| " + " | ".join(r) + " |" for r in rows]

    def write(self, path):
        path.write_text("\n".join(self.lines) + "\n")


def cell_table(md, rows, name, title, colspec, head, header, cellfn):
    """A table with one row per cell (d, t), the d blocks stacked: LaTeX into <name>.tex and a section of md."""
    tex, plain = [], []
    for d in DS:
        tex.append(f"\\multicolumn{{{colspec.count('r') + colspec.count('c')}}}{{l}}{{$d={d}$, $n={N}$}}\\\\")
        for t in TS:
            R = [r for r in rows if (r["d"], r["t"]) == (d, t)]
            tex.append(f"{t} & " + " & ".join(cellfn(R, True)) + "\\\\")
            plain.append([str(d), str(t)] + cellfn(R, False))
        if d != DS[-1]:
            tex.append("\\midrule")
    tabular(name, colspec, head, tex)
    md.table(title, ["d", "t"] + header, plain)


def col(R, field, digits, tex):
    """min/median/max of one field over the instances R."""
    return mmm([r[field] for r in R], digits, tex)


def grid_cells(R, tex):
    return [col(R, "prune_survivors", 0, tex), col(R, "prune_inliers_removed", 0, tex),
            col(R, "prune_outliers_left", 0, tex), col(R, "prune_seconds", 1, tex), col(R, "astar_surv_time", 2, tex),
            str(sum(r["astar_alone_run"] for r in R)), col(R, "astar_alone_time", 2, tex)]


def trimming_cells(R, tex):
    return [col(R, f"trim{k}_{f}", 2 if f == "astar_time" else 0, tex) for k in trimming.KS
            for f in ("survivors", "inliers_removed", "outliers_left", "astar_time")]


def wi_mbf_cells(R, tex):
    return [col(R, "m_star", 0, tex)] + [col(R, f"{pre}_{f}", 2 if f == "seconds" else 0, tex)
                                         for pre, _, _ in METHODS
                                         for f in ("survivors", "inliers_removed", "outliers_left", "seconds")]


def gore_cells(R, tex):
    G = [r for r in R if r["gore_survivors"] is not None]
    return [str(len(G)), col(G, "gore_survivors", 0, tex), col(G, "gore_inliers_removed", 0, tex),
            col(G, "gore_outliers_left", 0, tex), col(G, "gore_seconds", 1, tex),
            str(sum(bool(r["gore_at_limit"]) for r in G))]


def sweep_tables(md, rows):
    """Table 4: one row per sweep instance, k up to 5 in part a, from 10 in part b."""
    head = [" & & & & \\multicolumn{5}{c}{pruning and \\Astar{}} & \\multicolumn{2}{c}{bounds} & "
            "\\multicolumn{3}{c}{trimming at $2\\epsilon$} & \\multicolumn{4}{c}{WI} & "
            "\\multicolumn{4}{c}{MBF-MaxCon}\\\\",
            "\\cmidrule(lr){5-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-14}\\cmidrule(lr){15-18}"
            "\\cmidrule(lr){19-22}",
            f"$k$ & run & $m_0$ & $m^*$ & survivors & {IR} & {OL} & time & \\Astar{{}} time & lower & upper & "
            f"survivors & {IR} & {OL} & survivors & {IR} & {OL} & time & survivors & {IR} & {OL} & time\\\\"]
    header = ["k", "run", "m0", "m*", "survivors", "inliers removed", "outliers left", "time", "A* time", "lower",
              "upper", "trim survivors", "trim inliers removed", "trim outliers left"]
    header += [f"{m} {h}" for m in ("WI", "MBF") for h in ("survivors", "inliers removed", "outliers left", "time")]

    def cells(r, tex):
        c = [f"{r['k']:g}", str(r["run"])]
        c += [fmt(r[f], 0, tex) for f in ("m0", "m_star", "prune_survivors", "prune_inliers_removed",
                                          "prune_outliers_left")]
        c += [fmt(r["prune_seconds"], 1, tex), fmt(r["astar_surv_time"], 2, tex), fmt(r["prune_lower"], 0, tex),
              fmt(r["prune_upper"], 0, tex)]
        c += [fmt(r[f"trim2_{f}"], 0, tex) for f in ("survivors", "inliers_removed", "outliers_left")]
        for pre, _, _ in METHODS:
            c += [fmt(r[f"{pre}_{f}"], 0, tex) for f in ("survivors", "inliers_removed", "outliers_left")]
            c.append(fmt(r[f"{pre}_seconds"], 2, tex))
        return c

    plain, parts = [], {"a": [], "b": []}
    for k in sorted({r["k"] for r in rows}):
        part = parts["a" if k <= 5 else "b"]
        part.append(f"\\multicolumn{{22}}{{l}}{{$k={k:g}$, outlier residual in "
                    f"$(\\epsilon, {k:g}\\epsilon]$}}\\\\")
        for r in sorted((r for r in rows if r["k"] == k), key=lambda r: r["run"]):
            part.append(" & ".join(cells(r, True)) + "\\\\")
            plain.append(cells(r, False))
        part.append("\\midrule")
    for p in "ab":
        tabular(f"tab_sweep_{p}", "rrrr rrrrr rr rrr rrrr rrrr", head, parts[p][:-1], top=True)
    md.table("Table 4, the residual sweep at d = 8, t = 40, one row per instance", header, plain)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid, sweep = instance_rows(GRID, RES), instance_rows(SWEEP_NAMES, RES / "sweep")
    write_csv(OUT / "grid_instances.csv", grid)
    write_csv(OUT / "sweep_instances.csv", sweep)
    md = Markdown("Synthetic tables")
    cell_table(md, grid, "tab_grid", "Table 1, Algorithm 1 on the grid, per cell", "r rrrrr rr",
               [" & \\multicolumn{5}{c}{pruning and \\Astar{}} & \\multicolumn{2}{c}{\\Astar{} alone}\\\\",
                "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}",
                "$t$ & survivors & inliers removed & outliers left & time & \\Astar{} time & runs & time\\\\"],
               ["survivors", "inliers removed", "outliers left", "time", "A* time", "A* alone runs", "A* alone time"],
               grid_cells)
    ks = trimming.KS
    cell_table(md, grid, "tab_trimming", "Table 2, the trimming at k eps, per cell", "r" + " rrrr" * len(ks),
               [" & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{$k={k}$}}" for k in ks) + "\\\\",
                "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(ks))),
                "$t$ & " + " & ".join([f"survivors & {IR} & {OL} & \\Astar{{}} time"] * len(ks)) + "\\\\"],
               [f"k={k} {h}" for k in ks for h in ("survivors", "inliers removed", "outliers left", "A* time")],
               trimming_cells)
    cell_table(md, grid, "tab_wi_mbf", "Table 3, WI and MBF-MaxCon, per cell", "r r rrrr rrrr",
               [" & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{name}}}" for _, _, name in METHODS) + "\\\\",
                "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
                "$t$ & $m^*$ & " + " & ".join([f"survivors & {IR} & {OL} & time"] * len(METHODS)) + "\\\\"],
               ["m*"] + [f"{name} {h}" for _, _, name in METHODS
                         for h in ("survivors", "inliers removed", "outliers left", "time")], wi_mbf_cells)
    sweep_tables(md, sweep)
    cell_table(md, grid, "tab_gore", "Table 5, GORE at its published settings, per cell", "r rrrrrr",
               ["$t$ & runs & survivors & inliers removed & outliers left & time & at the limit\\\\"],
               ["runs", "survivors", "inliers removed", "outliers left", "time", "at the limit"], gore_cells)
    md.write(OUT / "tables.md")
    for name, rows in (("grid", grid), ("sweep", sweep)):
        print(f"{name}: {len(rows)} instances, pruned {sum(r['prune_survivors'] is not None for r in rows)}, "
              f"optimal set known {sum(r['m_star'] is not None for r in rows)}, A* alone lines "
              f"{sum(r['astar_alone_run'] for r in rows)}", flush=True)
    print(f"written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
