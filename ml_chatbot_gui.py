
# ─────────────────────────────────────────────────────────────────────────────
#  AI Data Assistant  —  v12  (Neo Tactile UI)
#  Neo Tactile colour scheme · pill buttons · inline graphs panel
# ─────────────────────────────────────────────────────────────────────────────
import json, re, threading, uuid, warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.tree  import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm   import SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ══════════════════════  NEO TACTILE PALETTE  ════════════════════════════════
C = {
    # backgrounds
    "bg"       : "#080d1a",
    "sidebar"  : "#0b1120",
    "panel"    : "#0e1525",
    "card"     : "#121c30",
    "cardHi"   : "#18253e",
    "border"   : "#1a2e4a",
    # accents
    "blue"     : "#2563eb",   # primary buttons
    "blueHi"   : "#3b82f6",   # hover
    "blueGlow" : "#1d4ed8",   # pressed / active
    "cyan"     : "#22d3ee",   # secondary highlight
    "green"    : "#10b981",
    "amber"    : "#f59e0b",
    "red"      : "#ef4444",
    # text
    "text"     : "#e8eeff",
    "muted"    : "#4e6280",
    "white"    : "#ffffff",
    # special
    "userBg"   : "#0f2044",
    "inputBg"  : "#0a1020",
    "navActive": "#2563eb",
    "navIdle"  : "#0e1525",
}

# chart palette — bright so they stand out on dark bg
CP = ["#3b82f6","#22d3ee","#10b981","#f59e0b","#f43f5e","#a78bfa","#34d399","#fbbf24"]

# ── chart helpers (consistent dark style) ─────────────────────────────────────
CH_BG   = "#0e1525"
CH_AX   = "#121c30"
CH_TEXT = "#e8eeff"
CH_GRID = "#1a2e4a"
CH_TICK = 10        # tick label size
CH_LBL  = 11        # axis label size
CH_TTL  = 12        # title size
CH_LW   = 2.2       # line width

def _ax_style(ax, title=""):
    ax.set_facecolor(CH_AX)
    ax.tick_params(colors=CH_TEXT, labelsize=CH_TICK, length=3, width=0.8)
    ax.xaxis.label.set_color(CH_TEXT); ax.xaxis.label.set_fontsize(CH_LBL)
    ax.yaxis.label.set_color(CH_TEXT); ax.yaxis.label.set_fontsize(CH_LBL)
    ax.set_title(title, color=CH_TEXT, fontsize=CH_TTL, pad=14, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(CH_GRID); sp.set_linewidth(0.8)
    ax.grid(color=CH_GRID, linestyle="--", linewidth=0.6, alpha=0.8)

def _embed(fig, parent, pady=6):
    c = FigureCanvasTkAgg(fig, master=parent)
    c.draw()
    w = c.get_tk_widget()
    w.configure(bg=CH_BG, highlightthickness=0)
    w.pack(fill=tk.BOTH, expand=True, padx=10, pady=pady)
    return c

# ══════════════════════  ML MODELS  ══════════════════════════════════════════
MODELS = {
    "1":{"name":"Random Forest",     "clf":lambda:RandomForestClassifier(n_estimators=80,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42),"reg":lambda:RandomForestRegressor(n_estimators=80,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42),"desc":"Great all-rounder"},
    "2":{"name":"Decision Tree",     "clf":lambda:DecisionTreeClassifier(max_depth=12,random_state=42),"reg":lambda:DecisionTreeRegressor(max_depth=12,random_state=42),"desc":"Simple & fast"},
    "3":{"name":"Gradient Boosting", "clf":lambda:GradientBoostingClassifier(n_estimators=80,random_state=42),"reg":lambda:GradientBoostingRegressor(n_estimators=80,random_state=42),"desc":"High accuracy"},
    "4":{"name":"Extra Trees",       "clf":lambda:ExtraTreesClassifier(n_estimators=80,n_jobs=-1,random_state=42),"reg":lambda:ExtraTreesRegressor(n_estimators=80,n_jobs=-1,random_state=42),"desc":"Faster ensemble"},
    "5":{"name":"AdaBoost",          "clf":lambda:AdaBoostClassifier(n_estimators=60,random_state=42),"reg":lambda:AdaBoostRegressor(n_estimators=60,random_state=42),"desc":"Boosting model"},
    "6":{"name":"KNN",               "clf":lambda:KNeighborsClassifier(n_neighbors=5,n_jobs=-1),"reg":lambda:KNeighborsRegressor(n_neighbors=5,n_jobs=-1),"desc":"Similarity based"},
    "7":{"name":"Logistic / Ridge",  "clf":lambda:LogisticRegression(max_iter=500,n_jobs=-1,random_state=42),"reg":lambda:Ridge(random_state=42),"desc":"Fast linear model"},
    "8":{"name":"SVM",               "clf":lambda:SVC(probability=True,random_state=42),"reg":lambda:SVR(),"desc":"Small clean data"},
}
ALIASES = {"random forest":"1","rf":"1","forest":"1","decision tree":"2","tree":"2","gradient boosting":"3","gb":"3","boosting":"3","extra trees":"4","et":"4","adaboost":"5","ada":"5","knn":"6","logistic":"7","ridge":"7","svm":"8","support vector":"8"}
SKIPS   = {"skip","dont know","don't know","idk","not sure","na","n/a","none","unknown","?","pass","ignore","empty","no idea"}

# ══════════════════════  SESSION  ════════════════════════════════════════════
class Session:
    def __init__(self):
        self.id=str(uuid.uuid4()); self.dataset=None; self.profile={}
        self.llm_summary=""; self.ollama_model="qwen2.5"
        self.task_type=None; self.target_column=None
        self.selected_feats=[]; self.ask_feats=[]; self.dropped_feats=[]
        self.null_strategy="fill"; self.chosen_model="1"
        self.model_bundle={}; self.metrics={}; self.feat_importance=[]
        self.col_defaults={}; self.stage="idle"; self.awaiting_feat=None
        self.pred_inputs={}; self.pred_result={}
        self.candidate_tgts=[]; self.id_cols=[]; self.note=""; self.top_k=5

# ══════════════════════  ML ENGINE  ══════════════════════════════════════════
class MLEngine:
    MAX=12000
    def train(self,df,target,features,null_strat,task_type,model_key="1"):
        data=df[features+[target]].copy()
        if null_strat=="drop": data=data.dropna()
        if len(data)<10: raise ValueError("Not enough rows (need ≥ 10).")
        if len(data)>self.MAX: data=data.sample(self.MAX,random_state=42)
        X=data[features].copy(); y=data[target].copy()
        cat=[c for c in features if X[c].dtype=="object" or str(X[c].dtype).startswith("category")]
        num=[c for c in features if c not in cat]
        for c in cat:
            X[c]=X[c].astype(str); keep=set(X[c].value_counts().head(30).index)
            X[c]=X[c].apply(lambda v:v if v in keep else "__OTHER__")
        pre=ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                             ("enc",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),cat)
        ])
        m=MODELS[model_key]; le=None
        if task_type=="classification":
            if y.dtype=="object" or str(y.dtype).startswith("category"):
                le=LabelEncoder(); y=le.fit_transform(y.astype(str))
            clf=m["clf"]()
        else:
            y=pd.to_numeric(y,errors="coerce"); mask=~pd.isna(y)
            X,y=X.loc[mask],y.loc[mask]; clf=m["reg"]()
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        pipe=Pipeline([("pre",pre),("clf",clf)]); pipe.fit(Xtr,ytr); preds=pipe.predict(Xte)
        if task_type=="classification":
            metrics={"Accuracy":round(float(accuracy_score(yte,preds)),4),
                     "F1":round(float(f1_score(yte,preds,average="weighted")),4)}
        else:
            metrics={"MAE":round(float(mean_absolute_error(yte,preds)),4),
                     "RMSE":round(float(np.sqrt(mean_squared_error(yte,preds))),4),
                     "R2":round(float(r2_score(yte,preds)),4)}
        names=pipe.named_steps["pre"].get_feature_names_out()
        imps=getattr(pipe.named_steps["clf"],"feature_importances_",np.ones(len(names))/len(names))
        # ── map sklearn prefixed names back to ORIGINAL column names ──────────
        orig_imp = {}
        for sk_name, imp_val in zip(names, imps):
            # sklearn names: "num__col_name" or "cat__col_name"
            raw = sk_name.split("__", 1)[1] if "__" in sk_name else sk_name
            raw_key = re.sub(r"[^a-z0-9]", "", raw.lower())
            matched = None
            # 1) exact match
            for c in features:
                if c == raw:
                    matched = c; break
            # 2) case-insensitive
            if not matched:
                for c in features:
                    if c.lower() == raw.lower():
                        matched = c; break
            # 3) fuzzy — strip non-alnum
            if not matched:
                for c in features:
                    if re.sub(r"[^a-z0-9]", "", c.lower()) == raw_key:
                        matched = c; break
            # 4) fallback — keep sklearn name
            if not matched:
                matched = raw
            orig_imp[matched] = orig_imp.get(matched, 0.0) + float(imp_val)
        fi = sorted([{"feature": k, "importance": v} for k, v in orig_imp.items()],
                    key=lambda x: x["importance"], reverse=True)[:10]
        return {"pipeline":pipe,"metrics":metrics,"fi":fi,"le":le,"task":task_type,
                "features":features,"target":target,"model_name":m["name"]}
    def predict(self,bundle,inputs):
        X=pd.DataFrame([inputs],columns=bundle["features"])
        pred=bundle["pipeline"].predict(X)[0]; prob=None
        if bundle["task"]=="classification":
            if hasattr(bundle["pipeline"].named_steps["clf"],"predict_proba"):
                prob=[round(float(v),4) for v in bundle["pipeline"].predict_proba(X)[0]]
            if bundle["le"] is not None: pred=bundle["le"].inverse_transform([int(pred)])[0]
        return {"prediction":pred,"probabilities":prob}

# ══════════════════════  OLLAMA  ═════════════════════════════════════════════
class Ollama:
    URL="http://localhost:11434"
    def ok(self):
        try: return requests.get(f"{self.URL}/api/tags",timeout=3).ok
        except: return False
    def ask(self,model,prompt):
        r=requests.post(f"{self.URL}/api/generate",json={"model":model,"prompt":prompt,"stream":False},timeout=120)
        r.raise_for_status(); return r.json().get("response","").strip()
    def summarize(self,model,profile):
        snap={"rows":profile["rows"],"columns":profile["columns"],"col_names":profile["col_names"][:20],
              "targets":profile["tgt_cols"][:5],"samples":{k:v for k,v in list(profile["samples"].items())[:8]}}
        return self.ask(model,"Summarize this dataset in 2-3 plain sentences. What is it about? Best column to predict?\n\n"+json.dumps(snap)[:2000])
    def explain(self,model,ctx):
        p=(f"Friendly ML explanation (≤220 words, no jargon):\n"
           f"Target:{ctx['target']} Predicted:{ctx['prediction']} Model:{ctx['model_name']}\n"
           f"Quality:{ctx['metrics']} TopFeatures:{ctx['top_features']}\n"
           f"UserGave:{ctx['user_inputs']} AutoFilled:{ctx['auto_filled']}\n"
           f"Explain:1)What predicted 2)Why(mention features) 3)How reliable 4)Auto-fill impact 5)Limits.")
        return self.ask(model,p[:3000])

# ══════════════════════  INLINE GRAPHS PANEL  ════════════════════════════════
class GraphsPanel(tk.Frame):
    """
    Inline graphs panel — shown when user clicks 📊 Graphs in sidebar.
    Draws all charts inside a scrollable canvas.
    Charts only render once prediction data is available.
    """
    def __init__(self, master):
        super().__init__(master, bg=C["bg"])
        self.graph_data = None   # set after prediction
        self._canvases = []
        self._build_shell()

    def _build_shell(self):
        # header
        hdr = tk.Frame(self, bg=C["panel"], height=44)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr, text="📊  Graphs & Statistical Summary",
                 bg=C["panel"], fg=C["text"], font=("Segoe UI",11,"bold")).pack(side=tk.LEFT,padx=16,pady=10)
        self.hdr_sub = tk.Label(hdr, text="Run a prediction first to see charts",
                                bg=C["panel"], fg=C["muted"], font=("Segoe UI",9))
        self.hdr_sub.pack(side=tk.LEFT, padx=4)
        # scrollable area
        outer = tk.Frame(self, bg=C["bg"])
        outer.pack(fill=tk.BOTH, expand=True)
        self.scroll_canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                          command=self.scroll_canvas.yview,
                          bg=C["panel"], troughcolor=C["bg"],
                          activebackground=C["blue"], width=8)
        self.scroll_canvas.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.scroll_canvas, bg=C["bg"])
        self._win = self.scroll_canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # placeholder
        self.placeholder = tk.Label(self.inner,
            text="📊  No data yet.\n\nComplete a prediction session first,\nthen click this tab to view all charts.",
            bg=C["bg"], fg=C["muted"], font=("Segoe UI",13), justify="center")
        self.placeholder.pack(expand=True, pady=120)

    def _on_inner_configure(self, e):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.scroll_canvas.itemconfig(self._win, width=e.width)

    def _on_mousewheel(self, e):
        self.scroll_canvas.yview_scroll(int(-1*(e.delta/120)), "units")

    def load(self, df, target, fi, task, metrics, model_name, pred_result):
        """Called after prediction — stores data and draws charts."""
        self.graph_data = dict(df=df,target=target,fi=fi,task=task,
                               metrics=metrics,model_name=model_name,pred_result=pred_result)
        # update subtitle
        pred = pred_result.get("prediction","—")
        try:   disp = f"{float(pred):,.2f}" if task=="regression" else str(pred)
        except: disp = str(pred)
        self.hdr_sub.config(text=f"Target: {target}   •   Predicted: {disp}   •   Model: {model_name}")
        self._draw()

    def _draw(self):
        """Clear old charts and redraw."""
        for w in self.inner.winfo_children():
            w.destroy()
        self._canvases.clear()
        d = self.graph_data
        self._section_fi(d["fi"])
        self._section_dist(d["df"], d["target"], d["task"])
        self._section_trends(d["df"], d["target"])
        self._section_stats(d["df"])

    # ── section label ─────────────────────────────────────────────────────────
    def _sec_label(self, text):
        row = tk.Frame(self.inner, bg=C["bg"])
        row.pack(fill=tk.X, padx=14, pady=(18,4))
        tk.Frame(row, bg=C["border"], height=1).pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(row, text=text, bg=C["bg"], fg=C["cyan"],
                 font=("Segoe UI",11,"bold")).pack(side=tk.LEFT)

    def _card(self, h=None):
        f = tk.Frame(self.inner, bg=C["card"], bd=0,
                     highlightbackground=C["border"], highlightthickness=1)
        f.pack(fill=tk.X, padx=14, pady=6)
        if h: f.configure(height=h)
        return f

    # ── Chart 1: Feature Importance ───────────────────────────────────────────
    def _section_fi(self, fi):
        self._sec_label("📈  Feature Importance  —  top columns driving the prediction")
        card = self._card()
        if not fi:
            tk.Label(card, text="No feature importance data available.",
                     bg=C["card"], fg=C["muted"]).pack(pady=20)
            return
        items  = fi[:8]
        names  = [x["feature"].split("__")[-1][:26] for x in items][::-1]
        vals   = [x["importance"] for x in items][::-1]
        colors = [CP[i % len(CP)] for i in range(len(names))]

        fig = Figure(figsize=(13, max(3.6, len(names)*0.52+1)), facecolor=CH_BG)
        ax  = fig.add_subplot(111)
        _ax_style(ax, "Feature Importance Score")
        bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.60)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.004,
                    bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", ha="left",
                    color=CH_TEXT, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=CH_LBL)
        ax.set_xlim(0, max(vals)*1.32)
        fig.tight_layout(pad=1.8)
        c = _embed(fig, card)
        self._canvases.append(c)

    # ── Chart 2: Distributions ────────────────────────────────────────────────
    def _section_dist(self, df, target, task):
        self._sec_label("📊  Distributions  —  target spread & numeric feature ranges")
        card = self._card()
        col      = df[target].dropna()
        num_cols = [c for c in df.columns
                    if c != target and str(df[c].dtype) in ("int64","float64")][:5]

        fig = Figure(figsize=(13, 5.2), facecolor=CH_BG)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # LEFT — target
        if task == "classification" or col.nunique() <= 20:
            vc = col.astype(str).value_counts().head(12)
            ax1.bar(range(len(vc)), vc.values, color=CP[0], edgecolor="none", width=0.65)
            ax1.set_xticks(range(len(vc)))
            ax1.set_xticklabels(vc.index, rotation=35, ha="right", fontsize=9)
            ax1.set_ylabel("Count", fontsize=CH_LBL)
        else:
            ax1.hist(col.astype(float), bins=30, color=CP[0],
                     edgecolor=CH_BG, linewidth=0.4, alpha=0.9)
            ax1.set_xlabel(target, fontsize=CH_LBL)
            ax1.set_ylabel("Frequency", fontsize=CH_LBL)
        _ax_style(ax1, f"Target  '{target}'  Distribution")

        # RIGHT — numeric overlays
        if num_cols:
            for i, c in enumerate(num_cols):
                d = df[c].dropna().astype(float)
                if d.std() > 0:
                    ax2.hist(d, bins=24, alpha=0.60, label=c,
                             color=CP[i % len(CP)], edgecolor=CH_BG, linewidth=0.3)
            ax2.set_xlabel("Value", fontsize=CH_LBL)
            ax2.set_ylabel("Frequency", fontsize=CH_LBL)
            leg = ax2.legend(fontsize=9, facecolor=C["card"],
                             edgecolor=C["border"], labelcolor=CH_TEXT, framealpha=0.9)
        else:
            ax2.set_visible(False)
        _ax_style(ax2, "Numeric Features  —  Value Spread")

        fig.tight_layout(pad=1.8)
        c = _embed(fig, card)
        self._canvases.append(c)

    # ── Chart 3: Trends ───────────────────────────────────────────────────────
    def _section_trends(self, df, target):
        self._sec_label("📉  Trends  —  relationships between top numeric features")
        card = self._card()
        num_cols = [c for c in df.columns
                    if c != target and str(df[c].dtype) in ("int64","float64")][:4]
        tgt_num  = str(df[target].dtype) in ("int64","float64")

        fig = Figure(figsize=(13, 5.2), facecolor=CH_BG)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # LEFT — scatter + trend
        if num_cols and tgt_num:
            best = num_cols[0]
            sdf  = df[[best,target]].dropna().sample(min(500,len(df)),random_state=42)
            xs   = sdf[best].astype(float).values
            ys   = sdf[target].astype(float).values
            ax1.scatter(xs, ys, alpha=0.50, color=CP[1], edgecolors="none", s=22)
            try:
                z  = np.polyfit(xs, ys, 1)
                xl = np.linspace(xs.min(), xs.max(), 150)
                ax1.plot(xl, np.polyval(z,xl), color=CP[3],
                         linewidth=CH_LW+0.4, linestyle="--", label="Trend")
                ax1.legend(fontsize=9, facecolor=C["card"],
                           edgecolor=C["border"], labelcolor=CH_TEXT)
            except: pass
            ax1.set_xlabel(best, fontsize=CH_LBL)
            ax1.set_ylabel(target, fontsize=CH_LBL)
            _ax_style(ax1, f"Scatter:  {best}  vs  {target}")
        else:
            ax1.set_visible(False)

        # RIGHT — line chart
        if len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            sdf2   = df[[c1,c2]].dropna().sort_values(c1).head(300)
            xv, yv = sdf2[c1].values, sdf2[c2].values
            ax2.plot(xv, yv, color=CP[2], linewidth=CH_LW)
            ax2.fill_between(xv, yv, alpha=0.15, color=CP[2])
            ax2.set_xlabel(c1, fontsize=CH_LBL); ax2.set_ylabel(c2, fontsize=CH_LBL)
            _ax_style(ax2, f"Line Trend:  {c1}  →  {c2}")
        elif num_cols:
            c1 = num_cols[0]; d = df[c1].dropna().astype(float).head(300)
            xv = np.arange(len(d)); yv = d.values
            ax2.plot(xv, yv, color=CP[2], linewidth=CH_LW)
            ax2.fill_between(xv, yv, alpha=0.15, color=CP[2])
            ax2.set_xlabel("Row index", fontsize=CH_LBL); ax2.set_ylabel(c1, fontsize=CH_LBL)
            _ax_style(ax2, f"Line Trend:  {c1}")
        else:
            ax2.set_visible(False)

        fig.tight_layout(pad=1.8)
        c = _embed(fig, card)
        self._canvases.append(c)

    # ── Section 4: Stats table ─────────────────────────────────────────────────
    def _section_stats(self, df):
        self._sec_label("🗂️  Statistical Summary  —  pandas describe()")
        card = self._card(h=260)

        try:   desc = df.describe(include="all").round(4).fillna("—")
        except: desc = df.describe().round(4)
        cols = list(desc.columns)

        frame = tk.Frame(card, bg=C["card"])
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)

        sy = tk.Scrollbar(frame, orient=tk.VERTICAL,   bg=C["card"], troughcolor=C["card"], activebackground=C["blue"], width=7)
        sx = tk.Scrollbar(frame, orient=tk.HORIZONTAL, bg=C["card"], troughcolor=C["card"], activebackground=C["blue"], width=7)
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")

        sty = ttk.Style()
        sty.configure("NT.Treeview",
                      background=C["card"], foreground=C["text"],
                      fieldbackground=C["card"], font=("Consolas",9), rowheight=24)
        sty.configure("NT.Treeview.Heading",
                      background=C["panel"], foreground=C["cyan"],
                      font=("Segoe UI",9,"bold"), relief="flat")
        sty.map("NT.Treeview",
                background=[("selected",C["blue"])],
                foreground=[("selected",C["white"])])

        tv = ttk.Treeview(frame, style="NT.Treeview",
                          columns=["Stat"]+cols, show="headings",
                          yscrollcommand=sy.set, xscrollcommand=sx.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sy.config(command=tv.yview); sx.config(command=tv.xview)

        tv.heading("Stat", text="Statistic"); tv.column("Stat", width=88, anchor="w", stretch=False)
        for c in cols:
            tv.heading(c, text=c); tv.column(c, width=105, anchor="center", stretch=False)

        tv.tag_configure("even", background=C["card"])
        tv.tag_configure("odd",  background=C["panel"])
        for i, (idx, row) in enumerate(desc.iterrows()):
            vals = [str(idx)] + [str(v) for v in row.values]
            tv.insert("","end", values=vals, tags=("even" if i%2==0 else "odd",))

# ══════════════════════  CONTROLLER  ═════════════════════════════════════════
class Controller:
    def __init__(self):
        self.s = Session(); self.ml = MLEngine(); self.llm = Ollama()
        self._done_cb = None

    def reset(self): self.s = Session()

    def _col(self, text):
        clean = re.sub(r"[^a-z0-9 ]","", text.strip().lower())
        if self.s.dataset is None: return None
        for c in self.s.dataset.columns:
            if re.sub(r"[^a-z0-9 ]","", c.strip().lower()) == clean: return c
        for c in self.s.dataset.columns:
            if clean and clean in re.sub(r"[^a-z0-9 ]","", c.strip().lower()): return c
        return None

    def _cols_in(self, text):
        found=[]; low=re.sub(r"[^a-z0-9 ]","",text.strip().lower())
        if self.s.dataset is None: return found
        for c in self.s.dataset.columns:
            k=re.sub(r"[^a-z0-9 ]","",c.strip().lower())
            if k and k in low: found.append(c)
        return found

    def _match_val(self, feat, text):
        col=self.s.dataset[feat]
        if str(col.dtype)=="object":
            raw=re.sub(r"[^a-z0-9 ]","",str(text).strip().lower()); lmap={}
            for v in col.dropna().astype(str).unique():
                k=re.sub(r"[^a-z0-9 ]","",v.strip().lower()); lmap.setdefault(k,v)
            if raw in lmap: return lmap[raw]
            for k,v in lmap.items():
                if raw and (raw in k or k in raw): return v
            return None
        try: return float(text)
        except: return None

    def _default(self, f):
        col=self.s.dataset[f].dropna()
        if str(col.dtype)=="object": vc=col.value_counts(); return vc.index[0] if len(vc)>0 else "unknown"
        return float(col.median()) if len(col)>0 else 0.0

    def _build_defaults(self):
        self.s.col_defaults = {f:self._default(f) for f in self.s.selected_feats}

    def _profile(self, df):
        return {"rows":int(df.shape[0]),"columns":int(df.shape[1]),
                "col_names":list(df.columns),
                "dtypes":{c:str(df[c].dtype) for c in df.columns},
                "nulls":{c:int(df[c].isna().sum()) for c in df.columns},
                "unique":{c:int(df[c].nunique(dropna=True)) for c in df.columns},
                "samples":{c:df[c].dropna().astype(str).head(3).tolist() for c in df.columns},
                "id_cols":self._det_ids(df),"tgt_cols":self._det_tgts(df)}

    def _det_ids(self, df):
        ids=[]; n=len(df)
        for c in df.columns:
            nm=c.strip().lower()
            if nm=="id" or nm.endswith("_id") or nm.endswith("id") or df[c].nunique()>=max(1,int(0.98*n)):
                ids.append(c)
        return ids

    def _det_tgts(self, df):
        prio=[]; rest=[]
        for c in df.columns:
            nm=re.sub(r"[^a-z]","",c.lower()); u=df[c].nunique(dropna=True)
            if nm in {"price","sellingprice","target","label","class","output","result"}: prio.insert(0,c)
            elif 2<=u<=min(30,max(2,len(df)//3)): rest.append(c)
        out=[]; seen=set()
        for c in prio+rest:
            if c not in seen: out.append(c); seen.add(c)
        return out or list(df.columns[-3:])

    def _guess_tgt(self, df):
        for c in df.columns:
            if re.sub(r"[^a-z]","",c.lower()) in {"price","sellingprice","target","label","class"}: return c
        t=self._det_tgts(df); return t[0] if t else df.columns[-1]

    def _infer_task(self, col):
        s=self.s.dataset[col]
        return "regression" if (str(s.dtype) not in {"object","category"} and s.nunique()>20) else "classification"

    def _train(self, note=""):
        feats=[c for c in self.s.dataset.columns if c!=self.s.target_column and c not in self.s.dropped_feats]
        if not feats: raise ValueError("No feature columns remain.")
        self.s.selected_feats=feats
        b=self.ml.train(self.s.dataset,self.s.target_column,feats,self.s.null_strategy,self.s.task_type,self.s.chosen_model)
        self.s.model_bundle=b; self.s.metrics=b["metrics"]; self.s.feat_importance=b["fi"]
        self.s.note=note; self._build_defaults(); self._upd_ask()

    def _topk(self):
        total=len(self.s.selected_feats)
        if total<=3: return total
        task=self.s.task_type or "regression"; fi=self.s.feat_importance
        n_cls=int(self.s.dataset[self.s.target_column].nunique(dropna=True))
        if task=="regression":   min_k,max_k=5,9
        elif n_cls==2:           min_k,max_k=3,5
        elif n_cls<=5:           min_k,max_k=4,7
        else:                    min_k,max_k=5,8
        max_k=min(max_k,total); min_k=min(min_k,total)
        if fi:
            tot=sum(x["importance"] for x in fi)
            if tot>0:
                cum=0.0; k=0
                for x in fi:
                    cum+=x["importance"]; k+=1
                    if cum/tot>=0.80: break
                return max(min_k,min(k,max_k))
        return min(max_k,max(min_k,5))

    def _upd_ask(self):
        tk_ = self._topk()
        # fi already stores ORIGINAL column names after the mapping in MLEngine.train()
        feat_set = set(self.s.selected_feats)
        # pick top-importance features that are valid column names
        top = [x["feature"] for x in self.s.feat_importance
               if x["feature"] in feat_set]
        seen = set()
        ask  = []
        for c in top:
            if c not in seen:
                ask.append(c); seen.add(c)
            if len(ask) >= tk_: break
        # fill remaining slots with next columns by dataset order
        for c in self.s.selected_feats:
            if len(ask) >= tk_: break
            if c not in seen:
                ask.append(c); seen.add(c)
        self.s.ask_feats = ask
        self.s.top_k     = tk_

    def load(self, path, model):
        self.reset(); self.s.ollama_model=model
        df=pd.read_csv(path)
        if df.empty or len(df.columns)<2: raise ValueError("CSV too small.")
        prof=self._profile(df); self.s.dataset=df; self.s.profile=prof
        self.s.id_cols=prof["id_cols"]; self.s.candidate_tgts=prof["tgt_cols"]
        self.s.target_column=self._guess_tgt(df); self.s.task_type=self._infer_task(self.s.target_column)
        self.s.dropped_feats=[]; self.s.stage="confirm_target"
        if self.llm.ok():
            try: self.s.llm_summary=self.llm.summarize(model,prof)
            except: self.s.llm_summary=self._fallback()
        else: self.s.llm_summary=self._fallback()
        self._train()
        q=self._ask()
        cols="  |  ".join(df.columns.tolist()[:15])+(" …" if len(df.columns)>15 else "")
        return (f"✅  Loaded  {prof['rows']:,} rows × {prof['columns']} columns\n\n"
                f"📋  Columns:  {cols}\n\n🤖  {self.s.llm_summary}\n\n"
                f"─────────────────────────────────\n\n{q}")

    def _fallback(self):
        return f"Dataset ready. Possible targets: {', '.join(self.s.candidate_tgts[:4]) or self.s.target_column}."

    def _mm(self):
        return "\n".join(f"  {k}.  {v['name']}  —  {v['desc']}" for k,v in MODELS.items())

    def _ask(self):
        s=self.s
        if s.stage=="confirm_target":
            opts="  |  ".join(s.candidate_tgts[:6]) if s.candidate_tgts else s.target_column
            return (f"🎯  What do you want to predict?\n\n"
                    f"Best guess:  **{s.target_column}**\n"
                    f"Options:  {opts}\n\n"
                    f"👉  Type a column name or reply  yes.")
        if s.stage=="confirm_ids":
            if s.id_cols:
                return (f"🆔  These look like IDs:\n   {'  |  '.join(s.id_cols[:6])}\n\n"
                        f"👉  yes  to ignore them,  no  to keep.")
            return "✅  No obvious ID columns.\n\n👉  Any column to ignore? Type a name or  no."
        if s.stage=="confirm_nulls":
            nulls=[k for k,v in s.profile.get("nulls",{}).items() if v>0]
            if nulls:
                return (f"🕳️  Columns with missing values:\n   {'  |  '.join(nulls[:6])}\n\n"
                        f"👉  fill  (use averages)   or   drop  (remove those rows)")
            return "✅  No missing values.\n\n👉  Reply  fill  or  drop."
        if s.stage=="confirm_features":
            top="  |  ".join(x["feature"] for x in s.feat_importance[:5]) or "—"
            return (f"🧩  Model trained!\n\nStrongest columns:\n   {top}\n\n"
                    f"👉  keep  to use all,  or  drop col1, col2  to remove some.")
        if s.stage=="choose_model":
            return f"🤖  Choose a ML model:\n\n{self._mm()}\n\n👉  Type number 1–8 or model name."
        if s.stage=="prediction": return self._next_q()
        return "What would you like to do next?"

    def reply(self, msg):
        if self.s.dataset is None: return "Please upload a CSV file first."
        low=msg.strip().lower(); st=self.s.stage
        if st=="confirm_target":
            if low not in {"yes","ok","y","sure","correct","yep"}:
                col=self._col(msg)
                if col: self.s.target_column=col; self.s.task_type=self._infer_task(col)
                else:
                    return ("❓  Could not find  '"+msg+"'.\n\nColumns:\n"
                            +"  |  ".join(self.s.dataset.columns.tolist())+"\n\nType one.")
            self._train("✅  Target confirmed. Task:  "+self.s.task_type)
            self.s.stage="confirm_ids"; return self._note()+self._ask()
        if st=="confirm_ids":
            if low in {"yes","ignore","y"}:
                self.s.dropped_feats=sorted(set(self.s.dropped_feats+self.s.id_cols))
            elif low not in {"no","keep","n"}:
                cols=self._cols_in(msg)
                if cols: self.s.dropped_feats=sorted(set(self.s.dropped_feats+cols))
            self._train("✅  Columns updated."); self.s.stage="confirm_nulls"
            return self._note()+self._ask()
        if st=="confirm_nulls":
            if any(w in low for w in ["fill","impute","average","mean"]): self.s.null_strategy="fill"
            elif any(w in low for w in ["drop","remove","delete"]): self.s.null_strategy="drop"
            self._train("✅  Missing-value strategy set."); self.s.stage="confirm_features"
            return self._note()+self._ask()
        if st=="confirm_features":
            if low not in {"keep","all","yes","ok","y"}:
                if any(w in low for w in ["drop","remove","ignore"]):
                    cols=self._cols_in(msg)
                    if cols: self.s.dropped_feats=sorted(set(self.s.dropped_feats+cols)); self._train("✅  Features updated.")
                    else: return "❓  Could not find those columns."
            self.s.stage="choose_model"; return self._note()+self._ask()
        if st=="choose_model":
            key=self._res_model(low)
            if not key: return f"❓  Did not recognise that.\n\n{self._mm()}"
            self.s.chosen_model=key; self._train(f"✅  Using  **{MODELS[key]['name']}**.")
            self.s.pred_inputs={}; self.s.stage="prediction"
            return (self._note()+
                    f"🚀  Ready!  Asking  **{self.s.top_k} key questions**  for  **{self.s.target_column}**.\n"
                    "   Type  skip  to auto-fill any question.\n\n"+self._ask())
        if st=="prediction": return self._handle_pred(msg)
        if st=="done": return "🏁  Done.  Click  📊 Graphs  in the sidebar to view all charts.\n\nPress  Reset  to start a new session."
        return "Something went wrong. Please reset."

    def _res_model(self, low):
        clean=re.sub(r"[^a-z0-9 ]","",low.strip())
        if clean in MODELS: return clean
        if clean in ALIASES: return ALIASES[clean]
        for alias,key in ALIASES.items():
            if alias in clean: return key
        return None

    def _note(self):
        n=self.s.note; self.s.note=""; return (n+"\n\n") if n else ""

    def _next_q(self):
        rem=[f for f in self.s.ask_feats if f not in self.s.pred_inputs]
        if not rem: return self._run_pred()
        feat=rem[0]; self.s.awaiting_feat=feat
        col=self.s.dataset[feat]; samp=col.dropna().astype(str).unique()[:4].tolist()
        total=len(self.s.ask_feats); done=len(self.s.pred_inputs)
        d=self.s.col_defaults.get(feat,"—"); ds=f"{d:,.2f}" if isinstance(d,float) else str(d)
        prog=f"[{done+1}/{total}]"
        if str(col.dtype)=="object":
            return (f"📝  {prog}  Value for  **{feat}**?\n\n"
                    f"Known values:  {' | '.join(samp)}\n"
                    f"💡  Type  skip  →  auto-fill  '{ds}'")
        return (f"🔢  {prog}  Value for  **{feat}**?\n\n"
                f"(Number like:  {samp[0] if samp else '0'})\n"
                f"💡  Type  skip  →  auto-fill  {ds}")

    def _handle_pred(self, msg):
        feat=self.s.awaiting_feat
        if not feat: return self._ask()
        low=re.sub(r"[^a-z0-9 ]","",msg.strip().lower())
        if low in SKIPS:
            d=self.s.col_defaults.get(feat); self.s.pred_inputs[feat]=d
            ds=f"{d:,.2f}" if isinstance(d,float) else str(d)
            return f"✅  Skipped  **{feat}**  →  using  **{ds}**\n\n"+self._next_q()
        val=self._match_val(feat,msg)
        if val is None:
            col=self.s.dataset[feat]
            if str(col.dtype)=="object":
                known=sorted(set(col.dropna().astype(str).tolist()))[:8]
                return f"❓  Did not recognise  '{msg}'  for  **{feat}**.\nChoose from:  {' | '.join(known)}\nOr  skip."
            return f"❓  **{feat}**  needs a number. Or  skip."
        self.s.pred_inputs[feat]=val; return self._next_q()

    def _run_pred(self):
        full={}; auto={}
        for f in self.s.selected_feats:
            if f in self.s.pred_inputs: full[f]=self.s.pred_inputs[f]
            else: d=self.s.col_defaults.get(f,0); full[f]=d; auto[f]=d
        result=self.ml.predict(self.s.model_bundle,full)
        self.s.pred_result=result; self.s.stage="done"
        pred=result["prediction"]; mname=self.s.model_bundle.get("model_name","ML")
        m=self.s.metrics; top5=self.s.feat_importance[:5]
        task=self.s.task_type; tgt=self.s.target_column
        top_str="  |  ".join(f"{x['feature']} ({x['importance']:.3f})" for x in top5) or "—"
        auto_str=", ".join(f"{k}={v}" for k,v in list(auto.items())[:5]) if auto else "none"
        if task=="classification":
            m_txt=f"Accuracy: {m.get('Accuracy')}   F1: {m.get('F1')}"
            det=(f"🎉  Prediction complete!\n\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"🏷️   **{tgt}**  →  **{pred}**\n"
                 f"🤖  Model:  {mname}\n📈  {m_txt}\n"
                 f"🔑  Top features:  {top_str}\n🔧  Auto-filled:  {auto_str}\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            try: pd_=f"{float(pred):,.2f}"
            except: pd_=str(pred)
            m_txt=f"MAE: {m.get('MAE')}   RMSE: {m.get('RMSE')}   R²: {m.get('R2')}"
            det=(f"🎉  Prediction complete!\n\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"💰  **{tgt}**  →  **{pd_}**\n"
                 f"🤖  Model:  {mname}\n📈  {m_txt}\n"
                 f"🔑  Top features:  {top_str}\n🔧  Auto-filled:  {auto_str}\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        llm_block=""
        if self.llm.ok():
            try:
                exp=self.llm.explain(self.s.ollama_model,{"target":tgt,"prediction":pred,"model_name":mname,"task_type":task,"metrics":m,"top_features":top5,"user_inputs":self.s.pred_inputs,"auto_filled":auto})
                llm_block=f"\n\n💬  In simple words:\n\n─────────────────────────\n{exp}\n─────────────────────────"
            except Exception as e: llm_block=f"\n\n⚠️  LLM explanation failed: {e}"
        else: llm_block="\n\n⚠️  Ollama not running."
        llm_block+="\n\n📊  Click  📊 Graphs  in the sidebar to see all charts & stats!"
        if self._done_cb:
            self._done_cb(self.s.dataset,tgt,self.s.feat_importance,task,m,mname,result)
        return det+llm_block

    def summary_json(self):
        s=self.s
        return json.dumps({
            "stage":s.stage,"target":s.target_column,"task":s.task_type,
            "model":MODELS.get(s.chosen_model,{}).get("name","—"),
            "features":len(s.selected_feats),"questions":s.top_k,
            "dropped":s.dropped_feats,"nulls":s.null_strategy,
            "metrics":s.metrics,"top5":s.feat_importance[:5],
            "inputs":s.pred_inputs,"result":s.pred_result
        },indent=2,default=str)

# ══════════════════════  MAIN APP  ═══════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root=root; self.ctrl=Controller()
        root.title("AI Data Assistant"); root.geometry("1540x940")
        root.configure(bg=C["bg"]); root.minsize(1200,760)
        self.ctrl._done_cb = self._on_prediction_done
        self._nav_state = "chat"
        self._build()
        self._welcome()

    # ── outer layout ──────────────────────────────────────────────────────────
    def _build(self):
        self._topbar()
        body=tk.Frame(self.root,bg=C["bg"]); body.pack(fill=tk.BOTH,expand=True)
        body.columnconfigure(0,weight=0); body.columnconfigure(1,weight=5); body.columnconfigure(2,weight=0)
        body.rowconfigure(0,weight=1)
        self._sidebar(body)
        # content area — chat and graphs frames share the same grid cell
        content=tk.Frame(body,bg=C["bg"]); content.grid(row=0,column=1,sticky="nsew")
        content.rowconfigure(0,weight=1); content.columnconfigure(0,weight=1)
        self._build_chat(content)
        self.graphs_panel=GraphsPanel(content); self.graphs_panel.grid(row=0,column=0,sticky="nsew")
        self._show_chat()   # chat is default
        self._right_panel(body)

    # ── top bar ───────────────────────────────────────────────────────────────
    def _topbar(self):
        bar=tk.Frame(self.root,bg=C["sidebar"],height=58)
        bar.pack(fill=tk.X); bar.pack_propagate(False)
        # logo
        tk.Label(bar,text="⚡",bg=C["sidebar"],fg=C["blue"],font=("Segoe UI",20,"bold")).pack(side=tk.LEFT,padx=(16,2),pady=8)
        tk.Label(bar,text="AI Data Assistant",bg=C["sidebar"],fg=C["white"],font=("Segoe UI",13,"bold")).pack(side=tk.LEFT,padx=(0,14))
        tk.Frame(bar,bg=C["border"],width=1,height=26).pack(side=tk.LEFT,fill=tk.Y,pady=16)
        tk.Label(bar,text="Upload CSV  →  Chat  →  Predict  →  Graphs",
                 bg=C["sidebar"],fg=C["muted"],font=("Segoe UI",9)).pack(side=tk.LEFT,padx=14)
        # right controls
        right=tk.Frame(bar,bg=C["sidebar"]); right.pack(side=tk.RIGHT,padx=14)
        tk.Label(right,text="Ollama:",bg=C["sidebar"],fg=C["muted"],font=("Segoe UI",9)).pack(side=tk.LEFT,padx=(0,4))
        self.model_var=tk.StringVar(value="qwen2.5")
        tk.Entry(right,textvariable=self.model_var,bg=C["card"],fg=C["text"],
                 insertbackground=C["cyan"],relief=tk.FLAT,font=("Segoe UI",10),width=14
                 ).pack(side=tk.LEFT,ipady=5,ipadx=6,padx=(0,10))
        self._pill_btn(right,"📂  Upload",self.upload,C["blue"]).pack(side=tk.LEFT,padx=3)
        self._pill_btn(right,"🔄  Reset", self.reset, C["card"]).pack(side=tk.LEFT,padx=3)
        self.status_var=tk.StringVar(value="Ready")
        tk.Label(bar,textvariable=self.status_var,bg=C["sidebar"],fg=C["cyan"],
                 font=("Segoe UI",9,"italic")).pack(side=tk.RIGHT,padx=20)

    def _pill_btn(self,parent,text,cmd,bg,fg=None):
        fg=fg or C["white"]
        b=tk.Label(parent,text=text,bg=bg,fg=fg,font=("Segoe UI",9,"bold"),
                   padx=14,pady=7,cursor="hand2")
        b.bind("<Button-1>",lambda e:cmd())
        b.bind("<Enter>",lambda e:b.config(bg=C["blueHi"] if bg==C["blue"] else C["blueGlow"]))
        b.bind("<Leave>",lambda e:b.config(bg=bg))
        return b

    # ── sidebar (only Chat + Graphs nav) ──────────────────────────────────────
    def _sidebar(self,parent):
        sb=tk.Frame(parent,bg=C["sidebar"],width=210)
        sb.grid(row=0,column=0,sticky="ns"); sb.pack_propagate(False)
        tk.Frame(sb,bg=C["border"],height=1).pack(fill=tk.X,pady=(10,8))
        # nav buttons
        self.nav_chat   = self._nav_item(sb,"💬","Chat",   lambda:self._show_chat())
        self.nav_graphs = self._nav_item(sb,"📊","Graphs", lambda:self._show_graphs())
        tk.Frame(sb,bg=C["border"],height=1).pack(fill=tk.X,pady=10)
        # session labels
        tk.Label(sb,text="SESSION",bg=C["sidebar"],fg=C["muted"],
                 font=("Segoe UI",8,"bold")).pack(anchor="w",padx=16,pady=(2,4))
        self.lbl_stage =self._sb_lbl(sb,"Stage: idle",  C["amber"])
        self.lbl_target=self._sb_lbl(sb,"Target: —",    C["muted"])
        self.lbl_model =self._sb_lbl(sb,"Model: —",     C["muted"])
        self.lbl_task  =self._sb_lbl(sb,"Task: —",      C["muted"])
        self.lbl_rows  =self._sb_lbl(sb,"Rows: —",      C["muted"])

    def _sb_lbl(self,parent,text,fg):
        lbl=tk.Label(parent,text=text,bg=C["sidebar"],fg=fg,
                     font=("Segoe UI",9),wraplength=185,justify="left")
        lbl.pack(anchor="w",padx=16,pady=1); return lbl

    def _nav_item(self,parent,icon,label,cmd):
        row=tk.Frame(parent,bg=C["navIdle"],cursor="hand2")
        row.pack(fill=tk.X,padx=8,pady=2)
        ico=tk.Label(row,text=icon,bg=C["navIdle"],fg=C["text"],font=("Segoe UI",13),padx=14,pady=10)
        ico.pack(side=tk.LEFT)
        lbl=tk.Label(row,text=label,bg=C["navIdle"],fg=C["text"],font=("Segoe UI",10,"bold"),pady=10)
        lbl.pack(side=tk.LEFT)
        for w in [row,ico,lbl]:
            w.bind("<Button-1>",lambda e,c=cmd:c())
        return {"row":row,"ico":ico,"lbl":lbl}

    def _set_nav_active(self, key):
        for nav,item in [("chat",self.nav_chat),("graphs",self.nav_graphs)]:
            bg = C["navActive"] if nav==key else C["navIdle"]
            fg = C["white"]     if nav==key else C["muted"]
            for w in [item["row"],item["ico"],item["lbl"]]:
                w.config(bg=bg)
            item["lbl"].config(fg=fg)
        self._nav_state=key

    def _show_chat(self):
        self.graphs_panel.grid_remove()
        self.chat_outer.grid(row=0,column=0,sticky="nsew")
        self._set_nav_active("chat")

    def _show_graphs(self):
        self.chat_outer.grid_remove()
        self.graphs_panel.grid(row=0,column=0,sticky="nsew")
        self._set_nav_active("graphs")

    # ── chat panel ────────────────────────────────────────────────────────────
    def _build_chat(self,parent):
        self.chat_outer=tk.Frame(parent,bg=C["bg"])
        self.chat_outer.grid(row=0,column=0,sticky="nsew")
        self.chat_outer.rowconfigure(1,weight=1); self.chat_outer.columnconfigure(0,weight=1)
        # header
        hdr=tk.Frame(self.chat_outer,bg=C["panel"],height=44)
        hdr.grid(row=0,column=0,sticky="ew"); hdr.pack_propagate(False)
        tk.Label(hdr,text="💬  Chat",bg=C["panel"],fg=C["text"],
                 font=("Segoe UI",11,"bold")).pack(side=tk.LEFT,padx=16,pady=10)
        self.chat_sub=tk.Label(hdr,text="Upload a CSV to begin",
                               bg=C["panel"],fg=C["muted"],font=("Segoe UI",9))
        self.chat_sub.pack(side=tk.LEFT,padx=4)
        # messages
        msg=tk.Frame(self.chat_outer,bg=C["bg"])
        msg.grid(row=1,column=0,sticky="nsew")
        msg.rowconfigure(0,weight=1); msg.columnconfigure(0,weight=1)
        self.chat=tk.Text(msg,wrap=tk.WORD,state=tk.DISABLED,
                          font=("Segoe UI",11),bg=C["bg"],fg=C["text"],
                          insertbackground=C["text"],relief=tk.FLAT,
                          padx=18,pady=14,spacing1=4,spacing3=4,
                          selectbackground=C["blue"])
        sc=tk.Scrollbar(msg,command=self.chat.yview,bg=C["panel"],troughcolor=C["bg"],
                        activebackground=C["blue"],width=7)
        self.chat.config(yscrollcommand=sc.set)
        sc.grid(row=0,column=1,sticky="ns"); self.chat.grid(row=0,column=0,sticky="nsew")
        self.chat.tag_config("bot", background=C["panel"],  foreground=C["text"],  lmargin1=14,lmargin2=14,rmargin=80,spacing1=9,spacing3=9)
        self.chat.tag_config("user",background=C["userBg"], foreground=C["white"], lmargin1=80,lmargin2=80,rmargin=14,spacing1=9,spacing3=9)
        self.chat.tag_config("sys", foreground=C["green"],   lmargin1=14,spacing1=5,spacing3=5)
        # input bar
        inp=tk.Frame(self.chat_outer,bg=C["panel"],height=60)
        inp.grid(row=2,column=0,sticky="ew"); inp.pack_propagate(False)
        inner=tk.Frame(inp,bg=C["inputBg"]); inner.pack(fill=tk.BOTH,expand=True,padx=12,pady=8)
        inner.columnconfigure(0,weight=1)
        self.input_var=tk.StringVar()
        self.entry=tk.Entry(inner,textvariable=self.input_var,bg=C["inputBg"],fg=C["text"],
                            insertbackground=C["cyan"],relief=tk.FLAT,font=("Segoe UI",11),bd=0)
        self.entry.grid(row=0,column=0,sticky="ew",ipady=7,ipadx=10)
        self.entry.bind("<Return>",lambda e:self.send())
        send=tk.Label(inner,text="  ↑  ",bg=C["blue"],fg=C["white"],
                      font=("Segoe UI",13,"bold"),padx=12,pady=5,cursor="hand2")
        send.grid(row=0,column=1,padx=(6,0))
        send.bind("<Button-1>",lambda e:self.send())
        send.bind("<Enter>",lambda e:send.config(bg=C["blueHi"]))
        send.bind("<Leave>",lambda e:send.config(bg=C["blue"]))

    # ── right session panel ────────────────────────────────────────────────────
    def _right_panel(self,parent):
        frame=tk.Frame(parent,bg=C["sidebar"],width=292)
        frame.grid(row=0,column=2,sticky="nsew"); frame.pack_propagate(False)
        hdr=tk.Frame(frame,bg=C["sidebar"],height=44)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr,text="Session State",bg=C["sidebar"],fg=C["text"],
                 font=("Segoe UI",11,"bold")).pack(side=tk.LEFT,padx=16,pady=10)
        self.summary=tk.Text(frame,wrap=tk.WORD,state=tk.DISABLED,
                             font=("Consolas",8),bg=C["sidebar"],fg=C["muted"],
                             relief=tk.FLAT,padx=10,pady=8)
        ss=tk.Scrollbar(frame,command=self.summary.yview,bg=C["sidebar"],
                        troughcolor=C["sidebar"],activebackground=C["blue"],width=6)
        self.summary.config(yscrollcommand=ss.set)
        ss.pack(side=tk.RIGHT,fill=tk.Y); self.summary.pack(fill=tk.BOTH,expand=True)
        self._ref_summary()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _append(self,tag,text):
        self.chat.config(state=tk.NORMAL)
        icon={"bot":"🤖  ","user":"👤  ","sys":"   "}.get(tag,"")
        self.chat.insert(tk.END,f"\n{icon}{text}\n",tag)
        self.chat.see(tk.END); self.chat.config(state=tk.DISABLED)

    def _ref_summary(self):
        txt=self.ctrl.summary_json()
        self.summary.config(state=tk.NORMAL); self.summary.delete("1.0",tk.END)
        self.summary.insert(tk.END,txt); self.summary.config(state=tk.DISABLED)
        s=self.ctrl.s
        self.lbl_stage.config(text=f"Stage: {s.stage}")
        self.lbl_target.config(text=f"Target: {s.target_column or '—'}")
        self.lbl_model.config(text=f"Model: {MODELS.get(s.chosen_model,{}).get('name','—')}")
        self.lbl_task.config(text=f"Task: {s.task_type or '—'}")
        rows=f"{len(s.dataset):,}" if s.dataset is not None else "—"
        self.lbl_rows.config(text=f"Rows: {rows}")

    def _welcome(self):
        self._append("bot",
            "Welcome! 👋  Upload a CSV file to start a prediction session.\n\n"
            "What I'll do:\n"
            "  1️⃣   Choose what to predict\n"
            "  2️⃣   Ignore ID / irrelevant columns\n"
            "  3️⃣   Handle missing values\n"
            "  4️⃣   Pick a ML model  (8 choices)\n"
            "  5️⃣   Ask only the most important questions\n"
            "  6️⃣   Give prediction + explanation\n"
            "  7️⃣   Click  📊 Graphs  to see charts & stats")

    # ── actions ───────────────────────────────────────────────────────────────
    def upload(self):
        path=filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path: return
        self.status_var.set("⏳  Analyzing…")
        fname=path.split("/")[-1].split("\\")[-1]
        self.chat_sub.config(text=fname)
        self._append("sys",f"Loading:  {fname}")
        self._show_chat()
        def task():
            try:
                resp=self.ctrl.load(path, self.model_var.get().strip() or "qwen2.5")
                self.root.after(0,lambda:self._append("bot",resp))
                self.root.after(0,self._ref_summary)
                self.root.after(0,lambda:self.status_var.set("⌨️  Waiting for reply"))
            except Exception as ex:
                self.root.after(0,lambda:messagebox.showerror("Load Error",str(ex)))
                self.root.after(0,lambda:self.status_var.set("❌  Error"))
        threading.Thread(target=task,daemon=True).start()

    def reset(self):
        self.ctrl.reset()
        self.chat.config(state=tk.NORMAL); self.chat.delete("1.0",tk.END); self.chat.config(state=tk.DISABLED)
        self._welcome(); self._ref_summary()
        self.status_var.set("Ready"); self.chat_sub.config(text="Upload a CSV to begin")
        self._show_chat()

    def send(self):
        msg=self.input_var.get().strip()
        if not msg: return
        self.input_var.set(""); self._append("user",msg); self.status_var.set("⏳  Thinking…")
        def task():
            try:
                resp=self.ctrl.reply(msg)
                self.root.after(0,lambda:self._append("bot",resp))
                self.root.after(0,self._ref_summary)
                self.root.after(0,lambda:self.status_var.set("⌨️  Waiting for reply"))
            except Exception as ex:
                self.root.after(0,lambda:self._append("bot",f"Something went wrong:\n{ex}"))
                self.root.after(0,lambda:self.status_var.set("❌  Error"))
        threading.Thread(target=task,daemon=True).start()

    def _on_prediction_done(self, df, target, fi, task, metrics, model_name, pred_result):
        """Called from background thread — load graph data on main thread."""
        def load_graphs():
            self.graphs_panel.load(df, target, fi, task, metrics, model_name, pred_result)
            # flash the Graphs nav button to let user know data is ready
            for _ in range(3):
                self.root.after(200*_,   lambda:self.nav_graphs["row"].config(bg=C["blue"]))
                self.root.after(200*_+100,lambda:self.nav_graphs["row"].config(bg=C["navIdle"]))
        self.root.after(400, load_graphs)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
