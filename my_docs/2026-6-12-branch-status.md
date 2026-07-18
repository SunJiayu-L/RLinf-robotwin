# 分支状态记录

日期：2026-06-12

## 当前分支

```text
branch: debug-from-dd552cc
HEAD: dd552cc907e5cb964c274bac2f0904cf6dc7c86b
commit: chore(robotwin): checkpoint evaluation experiments
```

该分支从 `dd552cc` 创建，用于从严格配对评测功能加入之前的实验快照继续
排查和修改。

后续两个提交当前不在该分支中：

```text
9f30e03 feat(robotwin): add paired native evaluation manifest
7b40992 chore(robotwin): add native paired eval job
```

## 已保存的修改

切换分支前，`main` 上的已跟踪文件修改已保存到：

```text
stash@{0}: On main: wip before branching from dd552cc
```

主要包含：

```text
my_docs/2026-6-11-rlinf-robotwin-eval-debug-plan.md
my_docs/2026-6-9-experiment-log.md
```

恢复时先切回需要恢复修改的分支，然后执行：

```bash
git stash apply stash@{0}
```

确认恢复正确后再执行 `git stash drop stash@{0}`，不要直接使用
`git stash pop`，以免发生冲突时难以回退。

## 未跟踪文件

以下文件没有放入 stash，切换分支后仍保留在工作区：

```text
.convert_robotwin47_pi05_to_pytorch.sh
.current_install_job
.fix_openpi_transformers.sh
.install_job.sh
.verify_openpi_robotwin.sh
slurm_logs/
```

这些文件不属于 `dd552cc`，提交前需要确认是否应加入版本控制。

