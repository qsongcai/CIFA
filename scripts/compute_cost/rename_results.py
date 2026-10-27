import json, glob, os
OUT = "/home/qsong/cifa_loader_test/results_loader"
for f in glob.glob(os.path.join(OUT, "*.json")):
    d = json.load(open(f))
    new = "{}_{}_{}_seed{}_valw{}.json".format(
        d["method"], d["variant"], d["target_domain"],
        d["seed"], d["val_workers"])
    newp = os.path.join(OUT, new)
    if os.path.abspath(f) == os.path.abspath(newp):
        print("skip(已是新名)", os.path.basename(f))
    else:
        os.rename(f, newp)
        print("rename", os.path.basename(f), "->", new)
