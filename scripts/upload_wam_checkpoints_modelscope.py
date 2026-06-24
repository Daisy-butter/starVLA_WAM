#!/usr/bin/env python3
"""Upload WAM LIBERO checkpoints to ModelScope (private)."""

import sys
import traceback
from datetime import datetime

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

TOKEN = "ms-c4c2fb50-ec00-4a5d-9290-41f0145b3bf6"
USERNAME = "DaisyButter"

UPLOADS = [
    {
        "repo_id": f"{USERNAME}/WAM-WanGR00T-libero_all-freeze_vae_umt5-train_dit_fmhead",
        "chinese_name": "WAM WanGR00T LIBERO-4in1, freeze VAE+UMT5, train DiT+FM head",
        "local_dir": "/SSD_DISK_1/users/wuruihan/WAM/work_dirs/libero/all/wan_gr00t/20260616_124043/checkpoints",
    },
    {
        "repo_id": f"{USERNAME}/WAM-WanOFT-libero_all-freeze_vae_umt5-train_dit_mlphead",
        "chinese_name": "WAM WanOFT LIBERO-4in1, freeze VAE+UMT5, train DiT+MLP head",
        "local_dir": "/SSD_DISK_1/users/wuruihan/WAM/work_dirs/libero/all/wan_oft/20260620_044639/checkpoints",
    },
    {
        "repo_id": f"{USERNAME}/WAM-WanGR00T-libero_spatial-freeze_backbone-train_fmhead",
        "chinese_name": "WAM WanGR00T LIBERO-spatial, freeze full backbone, train FM head",
        "local_dir": "/SSD_DISK_1/users/wuruihan/WAM/work_dirs/libero/spatial/wan_gr00t/20260615_100330/checkpoints",
    },
    {
        "repo_id": f"{USERNAME}/WAM-WanDit4DiT-libero_spatial-freeze_vae_umt5-train_videodit_actiondit",
        "chinese_name": "WAM WanDit4DiT LIBERO-spatial, freeze VAE+UMT5, train video+action DiT",
        "local_dir": "/SSD_DISK_1/users/wuruihan/WAM/work_dirs/libero/spatial/wan_dit4dit/20260622_134126/checkpoints",
    },
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def ensure_repo(api: HubApi, repo_id: str, chinese_name: str) -> None:
    try:
        api.get_model(repo_id)
        log(f"repo exists: {repo_id}")
    except Exception:
        log(f"creating repo: {repo_id}")
        api.create_model(
            model_id=repo_id,
            visibility=ModelVisibility.PRIVATE,
            license=Licenses.APACHE_V2,
            chinese_name=chinese_name,
        )
        log(f"created repo: {repo_id}")


def main() -> int:
    api = HubApi()
    api.login(TOKEN)
    log(f"logged in as {USERNAME}")

    failed = []
    for i, item in enumerate(UPLOADS, 1):
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        log(f"[{i}/{len(UPLOADS)}] start upload: {repo_id}")
        log(f"  local_dir={local_dir}")
        try:
            ensure_repo(api, repo_id, item["chinese_name"])
            api.upload_folder(
                repo_id=repo_id,
                folder_path=local_dir,
                commit_message="Upload WAM LIBERO checkpoint(s)",
            )
            log(f"[{i}/{len(UPLOADS)}] done: {repo_id}")
        except Exception as exc:
            log(f"[{i}/{len(UPLOADS)}] FAILED: {repo_id}: {exc}")
            traceback.print_exc()
            failed.append(repo_id)

    if failed:
        log(f"upload finished with failures: {failed}")
        return 1
    log("all uploads completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
