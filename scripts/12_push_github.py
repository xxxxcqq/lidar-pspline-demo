# -*- coding: utf-8 -*-
"""用 GitHub REST API 创建仓库并一次性上传项目文件（不需要本地 git）。令牌从 GH_TOKEN 环境变量读取。"""
import base64, json, os, sys, urllib.error, urllib.request

TOKEN = os.environ.get("GH_TOKEN", "").strip()
if not TOKEN:
    import getpass
    TOKEN = getpass.getpass("粘贴 GitHub PAT（输入不回显）: ").strip()
if not TOKEN:
    sys.exit("未提供令牌")

API = "https://api.github.com"
REPO = "lidar-pspline-demo"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".venv", "dist", "__pycache__", ".git", ".trae", ".vscode"}
SKIP_FILES = {"token.txt", "push_log.txt", ".repo_url"}
HEAD = {
    "Authorization": "Bearer " + TOKEN,
    "Accept": "application/vnd.github+json",
    "User-Agent": "lidar-pspline-demo-publisher",
    "X-GitHub-Api-Version": "2022-11-28",
}


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, headers=HEAD, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return {"__status__": e.code, "__body__": e.read().decode()[:300]}


user = call("GET", "/user")
if "__status__" in user:
    sys.exit("令牌无效或权限不足：%s %s" % (user["__status__"], user["__body__"]))
owner = user["login"]
print("登录身份:", owner)

repo = call("GET", f"/repos/{owner}/{REPO}")
if "__status__" in repo:
    created = call("POST", "/user/repos", {
        "name": REPO,
        "description": "LIDAR penalized spline reproduction (Dash + precomputed JSON)",
        "private": False,
        "auto_init": False,
        "has_issues": False,
        "has_wiki": False,
    })
    if "__status__" in created:
        sys.exit("创建仓库失败：%s %s" % (created["__status__"], created["__body__"]))
    repo = created
    print("已创建仓库:", repo["html_url"])
else:
    print("仓库已存在，将更新内容:", repo["html_url"])

# 收集文件
files = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel_dir = os.path.relpath(dirpath, ROOT)
    if rel_dir != "." and rel_dir.split(os.sep)[0] in SKIP_DIRS:
        continue
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn.endswith(".pyc") or fn in SKIP_FILES:
            continue
        rel = os.path.normpath(os.path.join(rel_dir, fn)).replace(os.sep, "/")
        if rel.startswith(".") and rel not in (".gitignore",):
            continue
        files.append(rel)

files.sort()
print("待上传文件数:", len(files))

# 已有分支则作为 base_tree，保留历史
base_commit = None
base_tree = None
ref = call("GET", f"/repos/{owner}/{REPO}/git/ref/heads/main")
if "__status__" not in ref and ref.get("object"):
    base_commit = ref["object"]["sha"]
    c = call("GET", f"/repos/{owner}/{REPO}/git/commits/{base_commit}")
    if "__status__" not in c:
        base_tree = c["tree"]["sha"]
        print("基于已有提交更新:", base_commit[:8])
else:
    # 空仓库：Git Data API 不可用，先用 contents API 建一个初始提交
    init = call("PUT", f"/repos/{owner}/{REPO}/contents/README.md", {
        "message": "init",
        "content": base64.b64encode(b"# LIDAR penalized spline demo\n").decode(),
    })
    if "__status__" in init:
        sys.exit("初始化仓库失败：%s %s" % (init["__status__"], init["__body__"]))
    print("已初始化空仓库: main")
    ref = call("GET", f"/repos/{owner}/{REPO}/git/ref/heads/main")
    base_commit = ref["object"]["sha"]
    c = call("GET", f"/repos/{owner}/{REPO}/git/commits/{base_commit}")
    base_tree = c["tree"]["sha"]

tree = []
for i, rel in enumerate(files, 1):
    with open(os.path.join(ROOT, rel), "rb") as f:
        content = base64.b64encode(f.read()).decode()
    blob = call("POST", f"/repos/{owner}/{REPO}/git/blobs",
                {"content": content, "encoding": "base64"})
    if "__status__" in blob:
        sys.exit("上传 %s 失败：%s %s" % (rel, blob["__status__"], blob["__body__"]))
    tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    print("  [%d/%d] %s" % (i, len(files), rel))

tree_payload = {"tree": tree}
if base_tree:
    tree_payload["base_tree"] = base_tree
new_tree = call("POST", f"/repos/{owner}/{REPO}/git/trees", tree_payload)
if "__status__" in new_tree:
    sys.exit("生成 tree 失败：%s %s" % (new_tree["__status__"], new_tree["__body__"]))

commit_payload = {"message": "LIDAR 惩罚样条复现：Dash 应用 + 预计算 JSON + 静态导出",
                  "tree": new_tree["sha"]}
if base_commit:
    commit_payload["parents"] = [base_commit]
commit = call("POST", f"/repos/{owner}/{REPO}/git/commits", commit_payload)
if "__status__" in commit:
    sys.exit("提交失败：%s %s" % (commit["__status__"], commit["__body__"]))

if base_commit:
    upd = call("PATCH", f"/repos/{owner}/{REPO}/git/refs/heads/main",
               {"sha": commit["sha"], "force": True})
else:
    upd = call("POST", f"/repos/{owner}/{REPO}/git/refs",
               {"ref": "refs/heads/main", "sha": commit["sha"]})
if "__status__" in upd:
    sys.exit("更新分支失败：%s %s" % (upd["__status__"], upd["__body__"]))

print("完成")
print("repo_url:", repo["html_url"])
print("commit:", commit["sha"][:8])

with open(os.path.join(ROOT, ".repo_url"), "w", encoding="utf-8") as f:
    f.write(repo["html_url"])
