// Markdown rendering for chat messages.
//
// Mirrors Open WebUI's MarkdownTokens / MarkdownInlineTokens / CodeBlock: the
// text goes through marked's lexer and each token is rendered by hand, rather
// than through marked's own HTML output, so spacing matches (a blank line
// between blocks becomes a `my-2` spacer, not paragraph margins). Model output
// is untrusted, so the result always passes through DOMPurify.

import { icon } from "./icons.js";

const { marked, DOMPurify, hljs } = window;

marked.use({ breaks: true, gfm: true });

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

// A textarea decodes entities without the HTML parser's habit of dropping
// leading whitespace, which would glue "**bold** text" into "boldtext".
const decoder = document.createElement("textarea");

function unescapeHtml(text) {
  decoder.innerHTML = String(text ?? "");
  return decoder.value;
}

// marked has already HTML-escaped inline text; normalise so nothing is doubled.
const inlineText = (text) => escapeHtml(unescapeHtml(text));

function renderInline(tokens) {
  return (tokens ?? [])
    .map((token) => {
      switch (token.type) {
        case "escape":
        case "text":
          return token.tokens ? renderInline(token.tokens) : inlineText(token.text);
        case "html":
          return /<br\s*\/?>/.test(token.text) ? "<br>" : escapeHtml(token.text);
        case "link": {
          const title = token.title ? ` title="${escapeHtml(token.title)}"` : "";
          const body = token.tokens ? renderInline(token.tokens) : escapeHtml(token.text);
          return `<a href="${escapeHtml(token.href)}" target="_blank" rel="nofollow"${title}>${body}</a>`;
        }
        case "image":
          return `<img src="${escapeHtml(token.href)}" alt="${escapeHtml(token.text)}">`;
        case "strong":
          return `<strong>${renderInline(token.tokens)}</strong>`;
        case "em":
          return `<em>${renderInline(token.tokens)}</em>`;
        case "del":
          return `<del>${renderInline(token.tokens)}</del>`;
        case "codespan":
          return `<code class="codespan">${inlineText(token.text)}</code>`;
        case "br":
          return "<br>";
        default:
          return "";
      }
    })
    .join("");
}

function highlight(code, lang) {
  try {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
    return hljs.highlightAuto(code).value;
  } catch {
    return escapeHtml(code);
  }
}

function renderCodeBlock(token) {
  const lang = (token.lang ?? "").trim().split(/\s+/)[0];
  const code = token.text ?? "";
  const lines = code.split("\n").length;
  return (
    `<div class="code-block">` +
    `<div class="cb-frame" dir="ltr">` +
    `<div class="cb-lang">${escapeHtml(lang)}</div>` +
    `<div class="cb-toolbar"><div class="cb-actions">` +
    `<button type="button" class="cb-btn" data-action="collapse-code">` +
    `<span class="cb-chevron">${icon("chevronUpDown", "size-3")}</span>` +
    `<span class="cb-collapse-label">Collapse</span></button>` +
    `<button type="button" class="cb-btn" data-action="copy-code">Copy</button>` +
    `</div></div>` +
    `<div class="cb-body"><div class="cb-pad"></div>` +
    `<pre class="hljs cb-pre"><code class="language-${escapeHtml(lang)}">${highlight(code, lang)}</code></pre>` +
    `<div class="cb-hidden"><span>${lines} hidden lines</span></div>` +
    `</div></div></div>`
  );
}

function renderTable(token) {
  const align = (i) => (token.align[i] ? ` style="text-align: ${token.align[i]}"` : "");
  const head = token.header
    .map(
      (cell, i) =>
        `<th scope="col"${align(i)}><div class="md-th"><div class="md-th-inner">` +
        `${renderInline(cell.tokens)}</div></div></th>`
    )
    .join("");
  const rows = token.rows
    .map((row, r) => {
      const last = r === token.rows.length - 1 ? " md-td-last" : "";
      const cells = row
        .map(
          (cell, i) =>
            `<td class="md-td${last}"${align(i)}><div class="md-td-inner">` +
            `${renderInline(cell.tokens)}</div></td>`
        )
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return (
    `<div class="md-table"><div class="md-table-scroll"><table>` +
    `<thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>`
  );
}

function renderList(token) {
  const items = token.items
    .map((item) => {
      const box = item.task
        ? `<input type="checkbox" disabled${item.checked ? " checked" : ""}>`
        : "";
      const body = renderBlocks(item.tokens, token.loose);
      return item.task && !token.ordered
        ? `<li class="md-task">${box}<div>${body}</div></li>`
        : `<li>${box}${body}</li>`;
    })
    .join("");
  return token.ordered
    ? `<ol start="${token.start || 1}" dir="auto">${items}</ol>`
    : `<ul dir="auto">${items}</ul>`;
}

function renderBlocks(tokens, top = true) {
  return (tokens ?? [])
    .map((token) => {
      switch (token.type) {
        case "hr":
          return '<hr class="md-hr">';
        case "heading":
          return `<h${token.depth} dir="auto">${renderInline(token.tokens)}</h${token.depth}>`;
        case "code":
          return token.raw.includes("```") || token.raw.includes("~~~")
            ? renderCodeBlock(token)
            : escapeHtml(token.text);
        case "table":
          return renderTable(token);
        case "blockquote":
          return `<blockquote dir="auto">${renderBlocks(token.tokens)}</blockquote>`;
        case "list":
          return renderList(token);
        case "html":
          return escapeHtml(token.text);
        case "paragraph":
          return `<p dir="auto">${renderInline(token.tokens)}</p>`;
        case "text": {
          const body = token.tokens ? renderInline(token.tokens) : inlineText(token.text);
          return top ? `<p>${body}</p>` : body;
        }
        case "space":
          return '<div class="md-space"></div>';
        default:
          return "";
      }
    })
    .join("");
}

/** Markdown to sanitised HTML, in Open WebUI's message markup. */
export function renderMarkdown(content) {
  const html = renderBlocks(marked.lexer(content ?? ""));
  return DOMPurify.sanitize(html, { ADD_ATTR: ["target"] });
}

/** Copy and collapse for code blocks; wired once with event delegation. */
export function bindCodeBlockActions(root, copyText) {
  root.addEventListener("click", (event) => {
    const button = event.target.closest(".cb-btn");
    if (!button || !root.contains(button)) return;
    const block = button.closest(".code-block");
    if (!block) return;
    if (button.dataset.action === "copy-code") {
      copyText(block.querySelector("pre code")?.textContent ?? "");
      button.textContent = "Copied";
      setTimeout(() => (button.textContent = "Copy"), 1000);
    } else if (button.dataset.action === "collapse-code") {
      const collapsed = block.classList.toggle("collapsed");
      button.querySelector(".cb-collapse-label").textContent = collapsed ? "Expand" : "Collapse";
    }
  });
}
