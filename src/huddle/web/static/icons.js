// SVG icons, path data as used by Open WebUI v0.6.30's icon components.
// The glyphs themselves come from Heroicons (MIT, github.com/tailwindlabs/heroicons)
// and Iconoir (MIT, github.com/iconoir-icons/iconoir).

const OUTLINE = 'fill="none" stroke="currentColor"';
const ROUND = 'stroke-linecap="round" stroke-linejoin="round"';

const ICONS = {
  pencilSquare: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10"/>`,
  },
  search: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"/>`,
  },
  workspace: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 0 0 2.25-2.25V6a2.25 2.25 0 0 0-2.25-2.25H6A2.25 2.25 0 0 0 3.75 6v2.25A2.25 2.25 0 0 0 6 10.5Zm0 9.75h2.25A2.25 2.25 0 0 0 10.5 18v-2.25a2.25 2.25 0 0 0-2.25-2.25H6a2.25 2.25 0 0 0-2.25 2.25V18A2.25 2.25 0 0 0 6 20.25Zm9.75-9.75H18a2.25 2.25 0 0 0 2.25-2.25V6A2.25 2.25 0 0 0 18 3.75h-2.25A2.25 2.25 0 0 0 13.5 6v2.25a2.25 2.25 0 0 0 2.25 2.25Z"/>`,
  },
  sidebar: {
    attrs: OUTLINE,
    body: `<rect x="3" y="3" width="18" height="18" rx="5" ry="5" ${ROUND}/><path d="M9.5 21V3" ${ROUND}/>`,
  },
  chatBubbleDotted: {
    attrs: OUTLINE,
    body: `<path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 13.8214 2.48697 15.5291 3.33782 17L2.5 21.5L7 20.6622C8.47087 21.513 10.1786 22 12 22Z" ${ROUND} stroke-dasharray="2.5 3.5"/>`,
  },
  chatBubbleDottedChecked: {
    attrs: OUTLINE,
    body: `<path d="M8 12L11 15L16 10" ${ROUND}/><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 13.8214 2.48697 15.5291 3.33782 17L2.5 21.5L7 20.6622C8.47087 21.513 10.1786 22 12 22Z" ${ROUND} stroke-dasharray="2.5 3.5"/>`,
  },
  chatCheck: {
    attrs: OUTLINE,
    body: `<path d="M8 12L11 15L16 10" ${ROUND}/><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 13.8214 2.48697 15.5291 3.33782 17L2.5 21.5L7 20.6622C8.47087 21.513 10.1786 22 12 22Z" ${ROUND}/>`,
  },
  adjustmentsHorizontal: {
    attrs: 'fill="currentColor" stroke="currentColor"',
    body: `<path ${ROUND} d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75"/>`,
  },
  chevronDown: { attrs: OUTLINE, body: `<path ${ROUND} d="m19.5 8.25-7.5 7.5-7.5-7.5"/>` },
  chevronUp: { attrs: OUTLINE, body: `<path ${ROUND} d="m4.5 15.75 7.5-7.5 7.5 7.5"/>` },
  chevronRight: { attrs: OUTLINE, body: `<path ${ROUND} d="m8.25 4.5 7.5 7.5-7.5 7.5"/>` },
  chevronUpDown: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M8.25 15 12 18.75 15.75 15m-7.5-6L12 5.25 15.75 9"/>`,
  },
  bolt: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="m3.75 13.5 10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z"/>`,
  },
  check: { attrs: OUTLINE, body: `<path ${ROUND} d="m4.5 12.75 6 6 9-13.5"/>` },
  xMark: {
    viewBox: '0 0 20 20',
    attrs: 'fill="currentColor"',
    body: '<path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"/>',
  },
  eyeSlash: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.451 10.451 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.498a10.522 10.522 0 0 1-4.293 5.774M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.243-4.243m4.242 4.242L9.88 9.88"/>`,
  },
  ellipsisDots: {
    // ChatItem's inline 16px "more" glyph (Heroicons micro ellipsis-horizontal).
    viewBox: '0 0 16 16',
    attrs: 'fill="currentColor"',
    body: '<path d="M2 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0ZM6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0ZM12.5 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/>',
  },
  pencil: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487zm0 0L19.5 7.125"/>`,
  },
  garbageBin: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"/>`,
  },
  clipboard: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184"/>`,
  },
  arrowPath: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"/>`,
  },
  arrowUp: {
    viewBox: '0 0 16 16',
    attrs: 'fill="currentColor"',
    body: '<path fill-rule="evenodd" d="M8 14a.75.75 0 0 1-.75-.75V4.56L4.03 7.78a.75.75 0 0 1-1.06-1.06l4.5-4.5a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1-1.06 1.06L8.75 4.56v8.69A.75.75 0 0 1 8 14Z" clip-rule="evenodd"/>',
  },
  stop: {
    attrs: 'fill="currentColor"',
    body: '<path fill-rule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm6-2.438c0-.724.588-1.312 1.313-1.312h4.874c.725 0 1.313.588 1.313 1.313v4.874c0 .725-.588 1.313-1.313 1.313H9.564a1.312 1.312 0 01-1.313-1.313V9.564z" clip-rule="evenodd"/>',
  },
  info: {
    attrs: OUTLINE,
    body: `<path ${ROUND} d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"/>`,
  },
  arrowDown: {
    viewBox: '0 0 20 20',
    attrs: 'fill="currentColor"',
    body: '<path fill-rule="evenodd" d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z" clip-rule="evenodd"/>',
  },
};

/** An icon as an SVG string. `cls` sizes it (e.g. "size-4.5"); `stroke` is the stroke width. */
export function icon(name, cls = "size-4", stroke = 1.5) {
  const def = ICONS[name];
  if (!def) throw new Error(`unknown icon: ${name}`);
  const viewBox = def.viewBox ?? "0 0 24 24";
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" ${def.attrs} ` +
    `stroke-width="${stroke}" class="icon ${cls}" aria-hidden="true">${def.body}</svg>`
  );
}
