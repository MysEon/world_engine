from __future__ import annotations

import argparse
import sys

from ..core.models import DirectorEvent
from ..infra.config import load_settings
from ..narrative.engine import WorldEngine

HELP = """
命令（输入后回车）：
  (直接回车)        推进一回合
  event <描述>      作为创世神注入一个事件
  event <描述> @角色1,@角色2   注入事件并指定直接感知的角色
  status            查看世界状态
  history           查看最近事件
  outline           根据事件流生成小说大纲
  write [all|N]     撰写章节（默认 all，可指定章节号）
  export [路径]      导出小说为 Markdown
  quit / exit       退出
""".strip()


def print_step(result) -> None:
    print(f"\n────── 第 {result.turn} 回合 ──────")
    print(result.narrative)


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stdin.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(description="World Engine CLI")
    parser.add_argument("--config", default="config.yaml", help="config file path")
    parser.add_argument("--preset", default=None, help="world preset YAML path")
    parser.add_argument("--world", default="default", help="world id (persistence key)")
    args = parser.parse_args()

    settings = load_settings(args.config)
    engine = WorldEngine(settings, args.world)
    engine.load_world(_load_preset(args.preset))
    world = engine.world

    print(f"世界「{world.name}」已加载")
    print(f"在场角色：{', '.join(c.name for c in world.characters)}")
    print(HELP)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not line:
            result = engine.step()
            print_step(result)
            continue

        if line in ("quit", "exit", "q"):
            print("再见。")
            break

        if line == "status":
            print(f"\n世界：{world.name}")
            print(f"回合：{world.turn}")
            for c in world.characters:
                print(f"  · {c.name}（{c.role}） 地点：{c.location}")
            continue

        if line == "history":
            for e in engine.store.events(engine.world_id, limit=12):
                tag = e.actor if e.actor else e.kind
                print(f"  [{e.turn}][{tag}] {e.content[:80]}")
            continue

        if line.startswith("event "):
            spec = line[len("event ") :].strip()
            content, _, targets_raw = spec.partition("@")
            targets = [t.strip() for t in targets_raw.split(",") if t.strip()]
            event = DirectorEvent(content=content.strip(), targets=targets)
            result = engine.step(director_event=event)
            print_step(result)
            continue

        if line == "outline":
            novel = engine.plan_novel()
            print(f"\n《{novel.title}》")
            if novel.arc:
                print(f"叙事弧线：{novel.arc}")
            for ch in novel.chapters:
                print(f"  第{ch.number}章 《{ch.title}》  (回合 {ch.start_turn}-{ch.end_turn})")
                print(f"      {ch.summary}")
            continue

        if line == "write" or line.startswith("write "):
            arg = line[len("write") :].strip()
            numbers = None if arg in ("", "all") else [int(n) for n in arg.split(",") if n.strip().isdigit()]
            novel = engine.write_novel(numbers)
            for ch in novel.chapters:
                mark = "✓" if ch.content else "·"
                print(f"  {mark} 第{ch.number}章 《{ch.title}》 {len(ch.content)}字")
            continue

        if line == "export" or line.startswith("export "):
            path = line[len("export") :].strip() or "novel.md"
            markdown = engine.export_novel()
            with open(path, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"已导出到 {path}")
            continue

        print("无法识别的命令。输入 help 查看更多。")
        if line == "help":
            print(HELP)


def _load_preset(path: str | None) -> dict:
    import yaml
    from pathlib import Path

    if path is None:
        path = "presets/tavern.yaml"
    p = Path(path)
    if not p.exists():
        print(f"错误：找不到预设文件 {p}", file=sys.stderr)
        raise SystemExit(1)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
