import json, glob, os
for f in sorted(glob.glob("/home/qsong/cifa_loader_test/results_loader/*.json")):
    d = json.load(open(f))
    print("{:42s} method={:4s} variant={:6s} valw={} swad_ckpt={:3d} acc={:.2f} wall={:.0f}s holdout_eval={:.1f}s".format(
        os.path.basename(f),
        d["method"], d["variant"], d["val_workers"],
        d["swad_n_checkpoints"], d["target_acc"] * 100,
        d["wall_time_sec"], d["holdout_eval_sec"]))
