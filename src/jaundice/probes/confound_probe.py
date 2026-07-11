"""Confound-proof probe: is the jaundice signal skin-localized bilirubin or a global illumination shortcut?
Classical CV only (PIL+numpy), no model training. Runs on all 760 images at low res."""
import numpy as np, glob, os
from PIL import Image

ROOT = "jaundicedataset3"
PATHS, Y = [], []
for cls, lab in [("jaundice",1),("normal",0)]:
    for p in glob.glob(os.path.join(ROOT, cls, "*.jpg")):
        PATHS.append(p); Y.append(lab)
Y = np.array(Y)
print(f"images: {len(PATHS)} | jaundice={Y.sum()} normal={(1-Y).sum()}")

def skin_mask(rgb):
    r,g,b = rgb[...,0],rgb[...,1],rgb[...,2]
    mx = rgb.max(-1); mn = rgb.min(-1)
    rule_rgb = (r>95)&(g>40)&(b>20)&((mx-mn)>15)&(np.abs(r.astype(int)-g.astype(int))>15)&(r>g)&(r>b)
    # YCbCr rule (broader, helps darker tones)
    y = 0.299*r+0.587*g+0.114*b
    cb = 128-0.168736*r-0.331264*g+0.5*b
    cr = 128+0.5*r-0.418688*g-0.081312*b
    rule_ycc = (cr>133)&(cr<180)&(cb>77)&(cb<128)&(y>60)
    return rule_rgb | rule_ycc

def grayworld(rgb):
    f = rgb.astype(np.float64)
    m = f.reshape(-1,3).mean(0)+1e-6
    g = m.mean()
    return np.clip(f*(g/m), 0, 255)

# collect per-image mean blue for several variants
cols = {"raw_whole":[], "skin":[], "nonskin":[], "gw_skin":[], "gw_nonskin":[], "skin_frac":[]}
for p in PATHS:
    im = Image.open(p).convert("RGB").resize((256,256))
    a = np.asarray(im)
    m = skin_mask(a)
    cols["skin_frac"].append(m.mean())
    cols["raw_whole"].append(a[...,2].mean())
    cols["skin"].append(a[...,2][m].mean() if m.sum()>50 else np.nan)
    cols["nonskin"].append(a[...,2][~m].mean() if (~m).sum()>50 else np.nan)
    gw = grayworld(a)
    cols["gw_skin"].append(gw[...,2][m].mean() if m.sum()>50 else np.nan)
    cols["gw_nonskin"].append(gw[...,2][~m].mean() if (~m).sum()>50 else np.nan)

def auc(scores, y):
    s = np.asarray(scores, float); ok = ~np.isnan(s)
    s, yy = s[ok], y[ok]
    # rank-based AUC
    order = s.argsort(); ranks = np.empty_like(order, float); ranks[order]=np.arange(1,len(s)+1)
    n1 = yy.sum(); n0 = len(yy)-n1
    if n1==0 or n0==0: return float("nan"), ok.sum()
    a = (ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
    return max(a,1-a), ok.sum()

print(f"\nmean skin-fraction of image: {np.nanmean(cols['skin_frac']):.2%}")
print("\n=== AUC of mean-BLUE by region (higher = predicts jaundice) ===")
print(f"{'variant':<14} {'AUC':>6}  {'n':>5}   interpretation")
labels = {
 "raw_whole":"CONTROL: reproduce global-color baseline",
 "skin":"skin pixels only (raw)",
 "nonskin":"NON-skin only -> if high, CONFOUND proven",
 "gw_skin":"skin after gray-world (illuminant removed)",
 "gw_nonskin":"non-skin after gray-world",
}
for k in ["raw_whole","skin","nonskin","gw_skin","gw_nonskin"]:
    a,n = auc(cols[k], Y)
    print(f"{k:<14} {a:>6.3f}  {n:>5}   {labels[k]}")
