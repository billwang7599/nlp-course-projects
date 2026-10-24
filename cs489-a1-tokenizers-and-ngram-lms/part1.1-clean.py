import argparse
import html
import re
from pathlib import Path


class TextCleaner:
    DROPPED_LINK_NAMESPACES = {
        "category", "cat", "تصنيف", "kategorie", "luokka", "kategória", "категория",
        "thể loại", "分類", "分类",
        "file", "image", "ملف", "datei", "bild", "tiedosto", "kuva", "fájl", "kép",
        "файл", "tập tin", "hình", "文件",
    }

    COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    REF_SELF_CLOSING_RE = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)
    REF_BLOCK_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
    HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
    HORIZONTAL_RULE_RE = re.compile(r"^-{4,}[ \t]*$", re.MULTILINE)
    MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")
    TEMPLATE_TOKEN_RE = re.compile(r"(?P<open>\{\{)|(?P<close>\}\})")
    MAX_TEMPLATE_LEN = 50_000
    MAX_TABLE_LEN = 100_000
    TABLE_TOKEN_RE = re.compile(r"(?P<open>^[ \t]*\{\|)|(?P<close>^[ \t]*\|\})", re.MULTILINE)
    ZH_VARIANT_RE = re.compile(r"-\{(.*?)\}-", re.DOTALL)
    ZH_ENTRY_RE = re.compile(r"(zh(?:-[A-Za-z]+)*)\s*:\s*(.*?)\s*(?=;\s*zh(?:-[A-Za-z]+)*\s*:|;?\s*$)", re.DOTALL)
    ZH_PRIORITY = ("zh-hans", "zh-cn", "zh-sg")
    EXTERNAL_LINK_RE = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:[ \t]+([^\]]*))?\]")
    BARE_URL_RE = re.compile(r"https?://[^\s<>\]\[|]+")
    QUOTES_RE = re.compile(r"'{2,}")
    HEADING_RE = re.compile(r"^=+[ \t]*(.*?)[ \t]*=+[ \t]*$", re.MULTILINE)
    LIST_MARKER_RE = re.compile(r"^[*#:;]+[ \t]*", re.MULTILINE)
    ENTITY_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
    SPACE_CHARS_RE = re.compile("[\u00a0\u2002\u2003\u2007\u2009\u200a\u202f]")
    INVISIBLE_CHARS_RE = re.compile("[\u00ad\u200b\u200e\u200f\ufeff]")
    SPACES_RE = re.compile(r"[ \t]+")
    BLANK_RUN_RE = re.compile(r"\n{3,}")
    NEWLINES_RE = re.compile(r"(?:[ \t]*\n)+")

    def __init__(self) -> None:
        self.steps = [
            self._strip_comments,
            self._strip_refs_and_tags,
            self._strip_templates,
            self._strip_tables,
            self._strip_rules_and_magic_words,
            self._resolve_zh_variants,
            self._resolve_wikilinks,
            self._strip_external_links,
            self._strip_quotes,
            self._strip_headings,
            self._strip_list_markers,
            self._decode_entities,
            self._normalize_whitespace,
        ]

    def clean(self, text) -> str:
        for step in self.steps:
            text = step(text)
        return text

    def _strip_comments(self, text: str) -> str:
        return self.COMMENT_RE.sub("", text)

    def _strip_refs_and_tags(self, text: str) -> str:
        text = self.REF_SELF_CLOSING_RE.sub("", text)
        text = self.REF_BLOCK_RE.sub("", text)
        return self.HTML_TAG_RE.sub("", text)

    @staticmethod
    def _strip_balanced(text: str, token_re: re.Pattern, max_len: int | None = None) -> str:
        out: list[str] = []
        stack: list[int] = []
        prev = 0
        for m in token_re.finditer(text):
            out.append(text[prev:m.start()])
            out.append(m.group())
            prev = m.end()
            if m.lastgroup == "open":
                stack.append(len(out) - 1)
            elif stack:
                i = stack.pop()
                if max_len is not None and sum(map(len, out[i:])) > max_len:
                    continue
                del out[i:]
                prev = TextCleaner._skip_blank_lines_if_whole_line(out, text, prev)
        out.append(text[prev:])
        return "".join(out)

    @staticmethod
    def _skip_blank_lines_if_whole_line(out: list[str], text: str, pos: int) -> int:
        for piece in reversed(out):
            if piece:
                if not piece.endswith("\n"):
                    return pos
                break
        m = TextCleaner.NEWLINES_RE.match(text, pos)
        return m.end() if m else pos

    def _strip_templates(self, text: str) -> str:
        return self._strip_balanced(text, self.TEMPLATE_TOKEN_RE, self.MAX_TEMPLATE_LEN)

    def _strip_tables(self, text: str) -> str:
        return self._strip_balanced(text, self.TABLE_TOKEN_RE, self.MAX_TABLE_LEN)

    def _strip_rules_and_magic_words(self, text: str) -> str:
        text = self.HORIZONTAL_RULE_RE.sub("", text)
        return self.MAGIC_WORD_RE.sub("", text)

    def _resolve_zh_variants(self, text: str) -> str:
        def pick(m: re.Match) -> str:
            body = m.group(1)
            head, sep, rest = body.partition("|")
            if sep and ":" not in head and re.fullmatch(r"[A-Za-z;\- ]*", head):
                body = rest
            entries = {k.lower(): v for k, v in self.ZH_ENTRY_RE.findall(body)}
            if not entries:
                return body
            for lang in self.ZH_PRIORITY:
                if lang in entries:
                    return entries[lang]
            return next(iter(entries.values()))

        return self.ZH_VARIANT_RE.sub(pick, text)

    def _resolve_wikilinks(self, text: str) -> str:
        out: list[str] = []
        stack: list[int] = []
        prev = 0
        for m in re.finditer(r"\[\[|\]\]", text):
            out.append(text[prev:m.start()])
            out.append(m.group())
            prev = m.end()
            if m.group() == "[[":
                stack.append(len(out) - 1)
            elif stack:
                i = stack.pop()
                content = "".join(out[i + 1:-1])
                if "\n" in content:
                    continue
                del out[i:]
                shown = self._link_text(content)
                out.append(shown)
                if not shown:
                    prev = self._skip_blank_lines_if_whole_line(out, text, prev)
        out.append(text[prev:])
        return "".join(out)

    @staticmethod
    def _link_text(content: str) -> str:
        target = content.split("|", 1)[0]
        if not target.startswith(":") and ":" in target:
            prefix = target.split(":", 1)[0].strip().lower().replace("_", " ")
            if prefix in TextCleaner.DROPPED_LINK_NAMESPACES or re.fullmatch(r"[a-z]{2,3}", prefix):
                return ""
        shown = content.rsplit("|", 1)[-1]
        return shown if shown else target

    def _strip_external_links(self, text: str) -> str:
        text = self.EXTERNAL_LINK_RE.sub(lambda m: m.group(1) or "", text)
        return self.BARE_URL_RE.sub("", text)

    def _strip_quotes(self, text: str) -> str:
        return self.QUOTES_RE.sub("", text)

    def _strip_headings(self, text: str) -> str:
        out: list[str] = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            m = self.HEADING_RE.match(lines[i])
            if not m:
                out.append(lines[i])
            else:
                if out and not out[-1].strip():
                    out.pop()
                out.append(m.group(1))
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    i += 1
            i += 1
        return "\n".join(out)

    def _strip_list_markers(self, text: str) -> str:
        return self.LIST_MARKER_RE.sub("", text)

    def _decode_entities(self, text: str) -> str:
        text = self.ENTITY_RE.sub(lambda m: html.unescape(m.group()), text)
        text = self.SPACE_CHARS_RE.sub(" ", text)
        return self.INVISIBLE_CHARS_RE.sub("", text)

    def _normalize_whitespace(self, text: str) -> str:
        text = self.SPACES_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = self.BLANK_RUN_RE.sub("\n\n", text).strip("\n")
        return text + "\n" if text else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Wikimedia dumps into plaintext.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cleaner = TextCleaner()
    for src in sorted(p for p in args.input.rglob("*") if p.is_file()):
        dst = args.output / src.relative_to(args.input)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cleaned = cleaner.clean(src.read_text(encoding="utf-8"))
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(cleaned)
        print(f"{src} -> {dst} ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
