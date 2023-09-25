---
title: A primer on Graph Neural Networks with Pytorch Geometric
seo_title: Colab Notebook training a simple Graph convolutional network for graph classification on Mutag dataset with pytorch geometric.
---
<html>
<head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>A_Primer_on_Graph_Neural_Networks_(Liverpool)</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script><script>
(function() {
  function addWidgetsRenderer() {
    var mimeElement = document.querySelector('script[type="application/vnd.jupyter.widget-view+json"]');
    var scriptElement = document.createElement('script');
    var widgetRendererSrc = 'https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js';
    var widgetState;

    // Fallback for older version:
    try {
      widgetState = mimeElement && JSON.parse(mimeElement.innerHTML);

      if (widgetState && (widgetState.version_major < 2 || !widgetState.version_major)) {
        widgetRendererSrc = 'https://unpkg.com/jupyter-js-widgets@*/dist/embed.js';
      }
    } catch(e) {}

    scriptElement.src = widgetRendererSrc;
    document.body.appendChild(scriptElement);
  }

  document.addEventListener('DOMContentLoaded', addWidgetsRenderer);
}());
</script>




<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>



<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/*
 * Webkit scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
  background: var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
  background: rgb(var(--jp-scrollbar-thumb-color));
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-right: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-bottom: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar */

[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
  background-color: transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid transparent;
  border-right: var(--jp-scrollbar-endpad) solid transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid transparent;
  border-bottom: var(--jp-scrollbar-endpad) solid transparent;
}

/*
 * Phosphor
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}


/* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}


/* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}


/* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}


/* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}


/* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}


/* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}


/* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}


/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}


/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}


/* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}


/* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}


/* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}


/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}


/* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}


/* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


/* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

@charset "UTF-8";
/*!

Copyright 2015-present Palantir Technologies, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.

*/
html{
  -webkit-box-sizing:border-box;
          box-sizing:border-box; }

*,
*::before,
*::after{
  -webkit-box-sizing:inherit;
          box-sizing:inherit; }

body{
  text-transform:none;
  line-height:1.28581;
  letter-spacing:0;
  font-size:14px;
  font-weight:400;
  color:#182026;
  font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }

p{
  margin-top:0;
  margin-bottom:10px; }

small{
  font-size:12px; }

strong{
  font-weight:600; }

::-moz-selection{
  background:rgba(125, 188, 255, 0.6); }

::selection{
  background:rgba(125, 188, 255, 0.6); }
.bp3-heading{
  color:#182026;
  font-weight:600;
  margin:0 0 10px;
  padding:0; }
  .bp3-dark .bp3-heading{
    color:#f5f8fa; }

h1.bp3-heading, .bp3-running-text h1{
  line-height:40px;
  font-size:36px; }

h2.bp3-heading, .bp3-running-text h2{
  line-height:32px;
  font-size:28px; }

h3.bp3-heading, .bp3-running-text h3{
  line-height:25px;
  font-size:22px; }

h4.bp3-heading, .bp3-running-text h4{
  line-height:21px;
  font-size:18px; }

h5.bp3-heading, .bp3-running-text h5{
  line-height:19px;
  font-size:16px; }

h6.bp3-heading, .bp3-running-text h6{
  line-height:16px;
  font-size:14px; }
.bp3-ui-text{
  text-transform:none;
  line-height:1.28581;
  letter-spacing:0;
  font-size:14px;
  font-weight:400; }

.bp3-monospace-text{
  text-transform:none;
  font-family:monospace; }

.bp3-text-muted{
  color:#5c7080; }
  .bp3-dark .bp3-text-muted{
    color:#a7b6c2; }

.bp3-text-disabled{
  color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-text-disabled{
    color:rgba(167, 182, 194, 0.6); }

.bp3-text-overflow-ellipsis{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal; }
.bp3-running-text{
  line-height:1.5;
  font-size:14px; }
  .bp3-running-text h1{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h1{
      color:#f5f8fa; }
  .bp3-running-text h2{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h2{
      color:#f5f8fa; }
  .bp3-running-text h3{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h3{
      color:#f5f8fa; }
  .bp3-running-text h4{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h4{
      color:#f5f8fa; }
  .bp3-running-text h5{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h5{
      color:#f5f8fa; }
  .bp3-running-text h6{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h6{
      color:#f5f8fa; }
  .bp3-running-text hr{
    margin:20px 0;
    border:none;
    border-bottom:1px solid rgba(16, 22, 26, 0.15); }
    .bp3-dark .bp3-running-text hr{
      border-color:rgba(255, 255, 255, 0.15); }
  .bp3-running-text p{
    margin:0 0 10px;
    padding:0; }

.bp3-text-large{
  font-size:16px; }

.bp3-text-small{
  font-size:12px; }
a{
  text-decoration:none;
  color:#106ba3; }
  a:hover{
    cursor:pointer;
    text-decoration:underline;
    color:#106ba3; }
  a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
    color:inherit; }
  a code,
  .bp3-dark a code{
    color:inherit; }
  .bp3-dark a,
  .bp3-dark a:hover{
    color:#48aff0; }
    .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
    .bp3-dark a:hover .bp3-icon,
    .bp3-dark a:hover .bp3-icon-standard,
    .bp3-dark a:hover .bp3-icon-large{
      color:inherit; }
.bp3-running-text code, .bp3-code{
  text-transform:none;
  font-family:monospace;
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
  background:rgba(255, 255, 255, 0.7);
  padding:2px 5px;
  color:#5c7080;
  font-size:smaller; }
  .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#a7b6c2; }
  .bp3-running-text a > code, a > .bp3-code{
    color:#137cbd; }
    .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
      color:inherit; }

.bp3-running-text pre, .bp3-code-block{
  text-transform:none;
  font-family:monospace;
  display:block;
  margin:10px 0;
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
  background:rgba(255, 255, 255, 0.7);
  padding:13px 15px 12px;
  line-height:1.4;
  color:#182026;
  font-size:13px;
  word-break:break-all;
  word-wrap:break-word; }
  .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa; }
  .bp3-running-text pre > code, .bp3-code-block > code{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none;
    padding:0;
    color:inherit;
    font-size:inherit; }

.bp3-running-text kbd, .bp3-key{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  min-width:24px;
  height:24px;
  padding:3px 6px;
  vertical-align:middle;
  line-height:24px;
  color:#5c7080;
  font-family:inherit;
  font-size:12px; }
  .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
    margin-right:5px; }
  .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
    background:#394b59;
    color:#a7b6c2; }
.bp3-running-text blockquote, .bp3-blockquote{
  margin:0 0 10px;
  border-left:solid 4px rgba(167, 182, 194, 0.5);
  padding:0 20px; }
  .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
    border-color:rgba(115, 134, 148, 0.5); }
.bp3-running-text ul,
.bp3-running-text ol, .bp3-list{
  margin:10px 0;
  padding-left:30px; }
  .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
    margin-bottom:5px; }
  .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
  .bp3-running-text ul ul,
  .bp3-running-text ol ul,
  .bp3-list ul{
    margin-top:5px; }

.bp3-list-unstyled{
  margin:0;
  padding:0;
  list-style:none; }
  .bp3-list-unstyled li{
    padding:0; }
.bp3-rtl{
  text-align:right; }

.bp3-dark{
  color:#f5f8fa; }

:focus{
  outline:rgba(19, 124, 189, 0.6) auto 2px;
  outline-offset:2px;
  -moz-outline-radius:6px; }

.bp3-focus-disabled :focus{
  outline:none !important; }
  .bp3-focus-disabled :focus ~ .bp3-control-indicator{
    outline:none !important; }

.bp3-alert{
  max-width:400px;
  padding:20px; }

.bp3-alert-body{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-alert-body .bp3-icon{
    margin-top:0;
    margin-right:20px;
    font-size:40px; }

.bp3-alert-footer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:reverse;
      -ms-flex-direction:row-reverse;
          flex-direction:row-reverse;
  margin-top:10px; }
  .bp3-alert-footer .bp3-button{
    margin-left:10px; }
.bp3-breadcrumbs{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:wrap;
      flex-wrap:wrap;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  margin:0;
  cursor:default;
  height:30px;
  padding:0;
  list-style:none; }
  .bp3-breadcrumbs > li{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center; }
    .bp3-breadcrumbs > li::after{
      display:block;
      margin:0 5px;
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 0 0-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 0 0 1.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
      width:16px;
      height:16px;
      content:""; }
    .bp3-breadcrumbs > li:last-of-type::after{
      display:none; }

.bp3-breadcrumb,
.bp3-breadcrumb-current,
.bp3-breadcrumbs-collapsed{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  font-size:16px; }

.bp3-breadcrumb,
.bp3-breadcrumbs-collapsed{
  color:#5c7080; }

.bp3-breadcrumb:hover{
  text-decoration:none; }

.bp3-breadcrumb.bp3-disabled{
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-breadcrumb .bp3-icon{
  margin-right:5px; }

.bp3-breadcrumb-current{
  color:inherit;
  font-weight:600; }
  .bp3-breadcrumb-current .bp3-input{
    vertical-align:baseline;
    font-size:inherit;
    font-weight:inherit; }

.bp3-breadcrumbs-collapsed{
  margin-right:2px;
  border:none;
  border-radius:3px;
  background:#ced9e0;
  cursor:pointer;
  padding:1px 5px;
  vertical-align:text-bottom; }
  .bp3-breadcrumbs-collapsed::before{
    display:block;
    background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
    width:16px;
    height:16px;
    content:""; }
  .bp3-breadcrumbs-collapsed:hover{
    background:#bfccd6;
    text-decoration:none;
    color:#182026; }

.bp3-dark .bp3-breadcrumb,
.bp3-dark .bp3-breadcrumbs-collapsed{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumbs > li::after{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumb.bp3-disabled{
  color:rgba(167, 182, 194, 0.6); }

.bp3-dark .bp3-breadcrumb-current{
  color:#f5f8fa; }

.bp3-dark .bp3-breadcrumbs-collapsed{
  background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-breadcrumbs-collapsed:hover{
    background:rgba(16, 22, 26, 0.6);
    color:#f5f8fa; }
.bp3-button{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  padding:5px 10px;
  vertical-align:middle;
  text-align:left;
  font-size:14px;
  min-width:30px;
  min-height:30px; }
  .bp3-button > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-button > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-button::before,
  .bp3-button > *{
    margin-right:7px; }
  .bp3-button:empty::before,
  .bp3-button > :last-child{
    margin-right:0; }
  .bp3-button:empty{
    padding:0 !important; }
  .bp3-button:disabled, .bp3-button.bp3-disabled{
    cursor:not-allowed; }
  .bp3-button.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button.bp3-align-right,
  .bp3-align-right .bp3-button{
    text-align:right; }
  .bp3-button.bp3-align-left,
  .bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-button:not([class*="bp3-intent-"]){
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    color:#182026; }
    .bp3-button:not([class*="bp3-intent-"]):hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      background-clip:padding-box;
      background-color:#ebf1f5; }
    .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#d8e1e8;
      background-image:none; }
    .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      outline:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-button.bp3-intent-primary{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#106ba3; }
    .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#0e5a8a;
      background-image:none; }
    .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(19, 124, 189, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-success{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#0f9960;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-success:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#0d8050; }
    .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#0a6640;
      background-image:none; }
    .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(15, 153, 96, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-warning{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#d9822b;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#bf7326; }
    .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#a66321;
      background-image:none; }
    .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(217, 130, 43, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-danger{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#db3737;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#c23030; }
    .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#a82a2a;
      background-image:none; }
    .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(219, 55, 55, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
    stroke:#ffffff; }
  .bp3-button.bp3-large,
  .bp3-large .bp3-button{
    min-width:40px;
    min-height:40px;
    padding:5px 15px;
    font-size:16px; }
    .bp3-button.bp3-large::before,
    .bp3-button.bp3-large > *,
    .bp3-large .bp3-button::before,
    .bp3-large .bp3-button > *{
      margin-right:10px; }
    .bp3-button.bp3-large:empty::before,
    .bp3-button.bp3-large > :last-child,
    .bp3-large .bp3-button:empty::before,
    .bp3-large .bp3-button > :last-child{
      margin-right:0; }
  .bp3-button.bp3-small,
  .bp3-small .bp3-button{
    min-width:24px;
    min-height:24px;
    padding:0 7px; }
  .bp3-button.bp3-loading{
    position:relative; }
    .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
      visibility:hidden; }
    .bp3-button.bp3-loading .bp3-button-spinner{
      position:absolute;
      margin:0; }
    .bp3-button.bp3-loading > :not(.bp3-button-spinner){
      visibility:hidden; }
  .bp3-button[class*="bp3-icon-"]::before{
    line-height:1;
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-weight:400;
    font-style:normal;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    color:#5c7080; }
  .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
    color:#5c7080; }
    .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
      margin-left:7px; }
  .bp3-button .bp3-icon:first-child:last-child,
  .bp3-button .bp3-spinner + .bp3-icon:last-child{
    margin:0 -7px; }
  .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#30404d; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#202b33;
      background-image:none; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
      color:#a7b6c2; }
  .bp3-dark .bp3-button[class*="bp3-intent-"]{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background-image:none;
      color:rgba(255, 255, 255, 0.3); }
    .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#8a9ba8; }
  .bp3-button:disabled::before,
  .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
  .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
  .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
    color:inherit !important; }
  .bp3-button.bp3-minimal{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none; }
    .bp3-button.bp3-minimal:hover{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(167, 182, 194, 0.3);
      text-decoration:none;
      color:#182026; }
    .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(115, 134, 148, 0.3);
      color:#182026; }
    .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
      background:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-minimal{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none; }
      .bp3-dark .bp3-button.bp3-minimal:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        cursor:not-allowed;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }

a.bp3-button{
  text-align:center;
  text-decoration:none;
  -webkit-transition:none;
  transition:none; }
  a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
    color:#182026; }
  a.bp3-button.bp3-disabled{
    color:rgba(92, 112, 128, 0.6); }

.bp3-button-text{
  -webkit-box-flex:0;
      -ms-flex:0 1 auto;
          flex:0 1 auto; }

.bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
.bp3-button-group.bp3-align-left .bp3-button-text,
.bp3-button-group.bp3-align-right .bp3-button-text{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto; }
.bp3-button-group{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex; }
  .bp3-button-group .bp3-button{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    position:relative;
    z-index:4; }
    .bp3-button-group .bp3-button:focus{
      z-index:5; }
    .bp3-button-group .bp3-button:hover{
      z-index:6; }
    .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
      z-index:7; }
    .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
      z-index:3; }
    .bp3-button-group .bp3-button[class*="bp3-intent-"]{
      z-index:9; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
        z-index:10; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
        z-index:11; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
        z-index:12; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
        z-index:8; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
    border-top-left-radius:0;
    border-bottom-left-radius:0; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:-1px;
    border-top-right-radius:0;
    border-bottom-right-radius:0; }
  .bp3-button-group.bp3-minimal .bp3-button{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none; }
    .bp3-button-group.bp3-minimal .bp3-button:hover{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(167, 182, 194, 0.3);
      text-decoration:none;
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(115, 134, 148, 0.3);
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
      background:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:inherit; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        cursor:not-allowed;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
      color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
      color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button-group .bp3-popover-wrapper,
  .bp3-button-group .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button-group .bp3-button.bp3-fill,
  .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    vertical-align:top; }
    .bp3-button-group.bp3-vertical.bp3-fill{
      width:unset;
      height:100%; }
    .bp3-button-group.bp3-vertical .bp3-button{
      margin-right:0 !important;
      width:100%; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
      border-radius:3px 3px 0 0; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
      border-radius:0 0 3px 3px; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-bottom:-1px; }
  .bp3-button-group.bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:1px; }
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
    margin-bottom:1px; }
.bp3-callout{
  line-height:1.5;
  font-size:14px;
  position:relative;
  border-radius:3px;
  background-color:rgba(138, 155, 168, 0.15);
  width:100%;
  padding:10px 12px 9px; }
  .bp3-callout[class*="bp3-icon-"]{
    padding-left:40px; }
    .bp3-callout[class*="bp3-icon-"]::before{
      line-height:1;
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-weight:400;
      font-style:normal;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      position:absolute;
      top:10px;
      left:10px;
      color:#5c7080; }
  .bp3-callout.bp3-callout-icon{
    padding-left:40px; }
    .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
      position:absolute;
      top:10px;
      left:10px;
      color:#5c7080; }
  .bp3-callout .bp3-heading{
    margin-top:0;
    margin-bottom:5px;
    line-height:20px; }
    .bp3-callout .bp3-heading:last-child{
      margin-bottom:0; }
  .bp3-dark .bp3-callout{
    background-color:rgba(138, 155, 168, 0.2); }
    .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
  .bp3-callout.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15); }
    .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-primary .bp3-heading{
      color:#106ba3; }
    .bp3-dark .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#48aff0; }
  .bp3-callout.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15); }
    .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-success .bp3-heading{
      color:#0d8050; }
    .bp3-dark .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
        color:#3dcc91; }
  .bp3-callout.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15); }
    .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-warning .bp3-heading{
      color:#bf7326; }
    .bp3-dark .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#ffb366; }
  .bp3-callout.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15); }
    .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-danger .bp3-heading{
      color:#c23030; }
    .bp3-dark .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#ff7373; }
  .bp3-running-text .bp3-callout{
    margin:20px 0; }
.bp3-card{
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
  background-color:#ffffff;
  padding:20px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-card.bp3-dark,
  .bp3-dark .bp3-card{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
    background-color:#30404d; }

.bp3-elevation-0{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  .bp3-elevation-0.bp3-dark,
  .bp3-dark .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-1{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-1.bp3-dark,
  .bp3-dark .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-elevation-2{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-2.bp3-dark,
  .bp3-dark .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }

.bp3-elevation-3{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-3.bp3-dark,
  .bp3-dark .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-elevation-4{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-4.bp3-dark,
  .bp3-dark .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:hover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  cursor:pointer; }
  .bp3-card.bp3-interactive:hover.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:active{
  opacity:0.9;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  -webkit-transition-duration:0;
          transition-duration:0; }
  .bp3-card.bp3-interactive:active.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-collapse{
  height:0;
  overflow-y:hidden;
  -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-collapse .bp3-collapse-body{
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
      display:none; }

.bp3-context-menu .bp3-popover-target{
  display:block; }

.bp3-context-menu-popover-target{
  position:fixed; }

.bp3-divider{
  margin:5px;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  border-bottom:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-dialog-container{
  opacity:1;
  -webkit-transform:scale(1);
          transform:scale(1);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:100%;
  min-height:100%;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5); }
  .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5);
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }

.bp3-dialog{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:30px 0;
  border-radius:6px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  background:#ebf1f5;
  width:500px;
  padding-bottom:20px;
  pointer-events:all;
  -webkit-user-select:text;
     -moz-user-select:text;
      -ms-user-select:text;
          user-select:text; }
  .bp3-dialog:focus{
    outline:0; }
  .bp3-dialog.bp3-dark,
  .bp3-dark .bp3-dialog{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    background:#293742;
    color:#f5f8fa; }

.bp3-dialog-header{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border-radius:6px 6px 0 0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  background:#ffffff;
  min-height:40px;
  padding-right:5px;
  padding-left:20px; }
  .bp3-dialog-header .bp3-icon-large,
  .bp3-dialog-header .bp3-icon{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px;
    color:#5c7080; }
  .bp3-dialog-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    margin:0;
    line-height:inherit; }
    .bp3-dialog-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-dialog-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
    background:#30404d; }
    .bp3-dark .bp3-dialog-header .bp3-icon-large,
    .bp3-dark .bp3-dialog-header .bp3-icon{
      color:#a7b6c2; }

.bp3-dialog-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  margin:20px;
  line-height:18px; }

.bp3-dialog-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  margin:0 20px; }

.bp3-dialog-footer-actions{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:end;
      -ms-flex-pack:end;
          justify-content:flex-end; }
  .bp3-dialog-footer-actions .bp3-button{
    margin-left:10px; }
.bp3-drawer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  padding:0; }
  .bp3-drawer:focus{
    outline:0; }
  .bp3-drawer.bp3-position-top{
    top:0;
    right:0;
    left:0;
    height:50%; }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%); }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-position-bottom{
    right:0;
    bottom:0;
    left:0;
    height:50%; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-position-left{
    top:0;
    bottom:0;
    left:0;
    width:50%; }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%); }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-position-right{
    top:0;
    right:0;
    bottom:0;
    width:50%; }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right):not(.bp3-vertical){
    top:0;
    right:0;
    bottom:0;
    width:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right).bp3-vertical{
    right:0;
    bottom:0;
    left:0;
    height:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-dark,
  .bp3-dark .bp3-drawer{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    background:#30404d;
    color:#f5f8fa; }

.bp3-drawer-header{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  position:relative;
  border-radius:0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  min-height:40px;
  padding:5px;
  padding-left:20px; }
  .bp3-drawer-header .bp3-icon-large,
  .bp3-drawer-header .bp3-icon{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px;
    color:#5c7080; }
  .bp3-drawer-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    margin:0;
    line-height:inherit; }
    .bp3-drawer-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-drawer-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-drawer-header .bp3-icon-large,
    .bp3-dark .bp3-drawer-header .bp3-icon{
      color:#a7b6c2; }

.bp3-drawer-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  overflow:auto;
  line-height:18px; }

.bp3-drawer-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  position:relative;
  -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
  padding:10px 20px; }
  .bp3-dark .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
.bp3-editable-text{
  display:inline-block;
  position:relative;
  cursor:text;
  max-width:100%;
  vertical-align:top;
  white-space:nowrap; }
  .bp3-editable-text::before{
    position:absolute;
    top:-3px;
    right:-3px;
    bottom:-3px;
    left:-3px;
    border-radius:3px;
    content:"";
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-editable-text.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
    background-color:#ffffff; }
  .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#137cbd; }
  .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
  .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#0f9960; }
  .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
  .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#d9822b; }
  .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
  .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#db3737; }
  .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
  .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
  .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background-color:rgba(16, 22, 26, 0.3); }
  .bp3-dark .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#48aff0; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
            box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#3dcc91; }
  .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
            box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#ffb366; }
  .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
            box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#ff7373; }
  .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
            box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-editable-text-input,
.bp3-editable-text-content{
  display:inherit;
  position:relative;
  min-width:inherit;
  max-width:inherit;
  vertical-align:top;
  text-transform:inherit;
  letter-spacing:inherit;
  color:inherit;
  font:inherit;
  resize:none; }

.bp3-editable-text-input{
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  background:none;
  width:100%;
  padding:0;
  white-space:pre-wrap; }
  .bp3-editable-text-input::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input:focus{
    outline:none; }
  .bp3-editable-text-input::-ms-clear{
    display:none; }

.bp3-editable-text-content{
  overflow:hidden;
  padding-right:2px;
  text-overflow:ellipsis;
  white-space:pre; }
  .bp3-editable-text-editing > .bp3-editable-text-content{
    position:absolute;
    left:0;
    visibility:hidden; }
  .bp3-editable-text-placeholder > .bp3-editable-text-content{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(167, 182, 194, 0.6); }

.bp3-editable-text.bp3-multiline{
  display:block; }
  .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
    overflow:auto;
    white-space:pre-wrap;
    word-wrap:break-word; }
.bp3-control-group{
  -webkit-transform:translateZ(0);
          transform:translateZ(0);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:stretch;
      -ms-flex-align:stretch;
          align-items:stretch; }
  .bp3-control-group > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select,
  .bp3-control-group .bp3-input,
  .bp3-control-group .bp3-select{
    position:relative; }
  .bp3-control-group .bp3-input{
    z-index:2;
    border-radius:inherit; }
    .bp3-control-group .bp3-input:focus{
      z-index:14;
      border-radius:3px; }
    .bp3-control-group .bp3-input[class*="bp3-intent"]{
      z-index:13; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
        z-index:15; }
    .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
      z-index:1; }
  .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
    z-index:13; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
      z-index:15; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select select,
  .bp3-control-group .bp3-select select{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    z-index:4;
    border-radius:inherit; }
    .bp3-control-group .bp3-button:focus,
    .bp3-control-group .bp3-html-select select:focus,
    .bp3-control-group .bp3-select select:focus{
      z-index:5; }
    .bp3-control-group .bp3-button:hover,
    .bp3-control-group .bp3-html-select select:hover,
    .bp3-control-group .bp3-select select:hover{
      z-index:6; }
    .bp3-control-group .bp3-button:active,
    .bp3-control-group .bp3-html-select select:active,
    .bp3-control-group .bp3-select select:active{
      z-index:7; }
    .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
    .bp3-control-group .bp3-html-select select[readonly],
    .bp3-control-group .bp3-html-select select:disabled,
    .bp3-control-group .bp3-html-select select.bp3-disabled,
    .bp3-control-group .bp3-select select[readonly],
    .bp3-control-group .bp3-select select:disabled,
    .bp3-control-group .bp3-select select.bp3-disabled{
      z-index:3; }
    .bp3-control-group .bp3-button[class*="bp3-intent"],
    .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
    .bp3-control-group .bp3-select select[class*="bp3-intent"]{
      z-index:9; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
        z-index:10; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
        z-index:11; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
        z-index:12; }
      .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
        z-index:8; }
  .bp3-control-group .bp3-input-group > .bp3-icon,
  .bp3-control-group .bp3-input-group > .bp3-button,
  .bp3-control-group .bp3-input-group > .bp3-input-action{
    z-index:16; }
  .bp3-control-group .bp3-select::after,
  .bp3-control-group .bp3-html-select::after,
  .bp3-control-group .bp3-select > .bp3-icon,
  .bp3-control-group .bp3-html-select > .bp3-icon{
    z-index:17; }
  .bp3-control-group:not(.bp3-vertical) > *{
    margin-right:-1px; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > *{
    margin-right:0; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
    margin-left:1px; }
  .bp3-control-group .bp3-popover-wrapper,
  .bp3-control-group .bp3-popover-target{
    border-radius:inherit; }
  .bp3-control-group > :first-child{
    border-radius:3px 0 0 3px; }
  .bp3-control-group > :last-child{
    margin-right:0;
    border-radius:0 3px 3px 0; }
  .bp3-control-group > :only-child{
    margin-right:0;
    border-radius:3px; }
  .bp3-control-group .bp3-input-group .bp3-button{
    border-radius:3px; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-control-group.bp3-vertical > *{
      margin-top:-1px; }
    .bp3-control-group.bp3-vertical > :first-child{
      margin-top:0;
      border-radius:3px 3px 0 0; }
    .bp3-control-group.bp3-vertical > :last-child{
      border-radius:0 0 3px 3px; }
.bp3-control{
  display:block;
  position:relative;
  margin-bottom:10px;
  cursor:pointer;
  text-transform:none; }
  .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
  .bp3-control:hover input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#106ba3; }
  .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background:#0e5a8a; }
  .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(19, 124, 189, 0.5); }
  .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#106ba3; }
  .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#0e5a8a; }
  .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(14, 90, 138, 0.5); }
  .bp3-control:not(.bp3-align-right){
    padding-left:26px; }
    .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-26px; }
  .bp3-control.bp3-align-right{
    padding-right:26px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      margin-right:-26px; }
  .bp3-control.bp3-disabled{
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-control.bp3-inline{
    display:inline-block;
    margin-right:20px; }
  .bp3-control input{
    position:absolute;
    top:0;
    left:0;
    opacity:0;
    z-index:-1; }
  .bp3-control .bp3-control-indicator{
    display:inline-block;
    position:relative;
    margin-top:-3px;
    margin-right:10px;
    border:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    cursor:pointer;
    width:1em;
    height:1em;
    vertical-align:middle;
    font-size:16px;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-control .bp3-control-indicator::before{
      display:block;
      width:1em;
      height:1em;
      content:""; }
  .bp3-control:hover .bp3-control-indicator{
    background-color:#ebf1f5; }
  .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background:#d8e1e8; }
  .bp3-control input:disabled ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(206, 217, 224, 0.5);
    cursor:not-allowed; }
  .bp3-control input:focus ~ .bp3-control-indicator{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  .bp3-control.bp3-align-right .bp3-control-indicator{
    float:right;
    margin-top:1px;
    margin-left:10px; }
  .bp3-control.bp3-large{
    font-size:16px; }
    .bp3-control.bp3-large:not(.bp3-align-right){
      padding-left:30px; }
      .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-30px; }
    .bp3-control.bp3-large.bp3-align-right{
      padding-right:30px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-30px; }
    .bp3-control.bp3-large .bp3-control-indicator{
      font-size:20px; }
    .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-top:0; }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
  .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#106ba3; }
  .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background:#0e5a8a; }
  .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(19, 124, 189, 0.5); }
  .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#106ba3; }
  .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(14, 90, 138, 0.5); }
  .bp3-control.bp3-checkbox .bp3-control-indicator{
    border-radius:3px; }
  .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 0 0-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0 0 12 5z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-radio .bp3-control-indicator{
    border-radius:50%; }
  .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
    background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
  .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
    opacity:0.5; }
  .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
    -moz-outline-radius:16px; }
  .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(167, 182, 194, 0.5); }
  .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(115, 134, 148, 0.5); }
  .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(92, 112, 128, 0.5); }
  .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5); }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch:not(.bp3-align-right){
    padding-left:38px; }
    .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-38px; }
  .bp3-control.bp3-switch.bp3-align-right{
    padding-right:38px; }
    .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
      margin-right:-38px; }
  .bp3-control.bp3-switch .bp3-control-indicator{
    border:none;
    border-radius:1.75em;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    width:auto;
    min-width:1.75em;
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-control.bp3-switch .bp3-control-indicator::before{
      position:absolute;
      left:0;
      margin:2px;
      border-radius:50%;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      background:#ffffff;
      width:calc(1em - 4px);
      height:calc(1em - 4px);
      -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    left:calc(100% - 1em); }
  .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
    padding-left:45px; }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-45px; }
  .bp3-control.bp3-switch.bp3-large.bp3-align-right{
    padding-right:45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-right:-45px; }
  .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.7); }
  .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.9); }
  .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(57, 75, 89, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background:#394b59; }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-control.bp3-switch .bp3-switch-inner-text{
    text-align:center;
    font-size:0.7em; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
    visibility:hidden;
    margin-right:1.2em;
    margin-left:0.5em;
    line-height:0; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
    visibility:visible;
    margin-right:0.5em;
    margin-left:1.2em;
    line-height:1em; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
    visibility:visible;
    line-height:1em; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
    visibility:hidden;
    line-height:0; }
  .bp3-dark .bp3-control{
    color:#f5f8fa; }
    .bp3-dark .bp3-control.bp3-disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-control .bp3-control-indicator{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0)); }
    .bp3-dark .bp3-control:hover .bp3-control-indicator{
      background-color:#30404d; }
    .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background:#202b33; }
    .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      cursor:not-allowed; }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      color:rgba(167, 182, 194, 0.6); }
.bp3-file-input{
  display:inline-block;
  position:relative;
  cursor:pointer;
  height:30px; }
  .bp3-file-input input{
    opacity:0;
    margin:0;
    min-width:200px; }
    .bp3-file-input input:disabled + .bp3-file-upload-input,
    .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(206, 217, 224, 0.5);
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6);
      resize:none; }
      .bp3-file-input input:disabled + .bp3-file-upload-input::after,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
        outline:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        cursor:not-allowed;
        color:rgba(92, 112, 128, 0.6); }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
      .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:rgba(57, 75, 89, 0.5);
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          -webkit-box-shadow:none;
                  box-shadow:none;
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
  .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#182026; }
  .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#f5f8fa; }
  .bp3-file-input.bp3-fill{
    width:100%; }
  .bp3-file-input.bp3-large,
  .bp3-large .bp3-file-input{
    height:40px; }
  .bp3-file-input .bp3-file-upload-input-custom-text::after{
    content:attr(bp3-button-text); }

.bp3-file-upload-input{
  outline:none;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  height:30px;
  padding:0 10px;
  vertical-align:middle;
  line-height:30px;
  color:#182026;
  font-size:14px;
  font-weight:400;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  position:absolute;
  top:0;
  right:0;
  left:0;
  padding-right:80px;
  color:rgba(92, 112, 128, 0.6);
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-file-upload-input::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-file-upload-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(206, 217, 224, 0.5);
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6);
    resize:none; }
  .bp3-file-upload-input::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    color:#182026;
    min-width:24px;
    min-height:24px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    position:absolute;
    top:0;
    right:0;
    margin:3px;
    border-radius:3px;
    width:70px;
    text-align:center;
    line-height:24px;
    content:"Browse"; }
    .bp3-file-upload-input::after:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      background-clip:padding-box;
      background-color:#ebf1f5; }
    .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#d8e1e8;
      background-image:none; }
    .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
      outline:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-file-upload-input:hover::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5; }
  .bp3-file-upload-input:active::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none; }
  .bp3-large .bp3-file-upload-input{
    height:40px;
    line-height:40px;
    font-size:16px;
    padding-right:95px; }
    .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
      padding:0 15px; }
    .bp3-large .bp3-file-upload-input::after{
      min-width:30px;
      min-height:30px;
      margin:5px;
      width:85px;
      line-height:30px; }
  .bp3-dark .bp3-file-upload-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::after{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
        background-color:#30404d; }
      .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
        background-color:#202b33;
        background-image:none; }
      .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none;
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
    .bp3-dark .bp3-file-upload-input:hover::after{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#30404d; }
    .bp3-dark .bp3-file-upload-input:active::after{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#202b33;
      background-image:none; }

.bp3-file-upload-input::after{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
.bp3-form-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0 0 15px; }
  .bp3-form-group label.bp3-label{
    margin-bottom:5px; }
  .bp3-form-group .bp3-control{
    margin-top:7px; }
  .bp3-form-group .bp3-form-helper-text{
    margin-top:5px;
    color:#5c7080;
    font-size:12px; }
  .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#106ba3; }
  .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#0d8050; }
  .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#bf7326; }
  .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#c23030; }
  .bp3-form-group.bp3-inline{
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
    .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
      margin:0 10px 0 0;
      line-height:40px; }
    .bp3-form-group.bp3-inline label.bp3-label{
      margin:0 10px 0 0;
      line-height:30px; }
  .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#48aff0; }
  .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#3dcc91; }
  .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#ffb366; }
  .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#ff7373; }
  .bp3-dark .bp3-form-group .bp3-form-helper-text{
    color:#a7b6c2; }
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(167, 182, 194, 0.6) !important; }
.bp3-input-group{
  display:block;
  position:relative; }
  .bp3-input-group .bp3-input{
    position:relative;
    width:100%; }
    .bp3-input-group .bp3-input:not(:first-child){
      padding-left:30px; }
    .bp3-input-group .bp3-input:not(:last-child){
      padding-right:30px; }
  .bp3-input-group .bp3-input-action,
  .bp3-input-group > .bp3-button,
  .bp3-input-group > .bp3-icon{
    position:absolute;
    top:0; }
    .bp3-input-group .bp3-input-action:first-child,
    .bp3-input-group > .bp3-button:first-child,
    .bp3-input-group > .bp3-icon:first-child{
      left:0; }
    .bp3-input-group .bp3-input-action:last-child,
    .bp3-input-group > .bp3-button:last-child,
    .bp3-input-group > .bp3-icon:last-child{
      right:0; }
  .bp3-input-group .bp3-button{
    min-width:24px;
    min-height:24px;
    margin:3px;
    padding:0 7px; }
    .bp3-input-group .bp3-button:empty{
      padding:0; }
  .bp3-input-group > .bp3-icon{
    z-index:1;
    color:#5c7080; }
    .bp3-input-group > .bp3-icon:empty{
      line-height:1;
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-weight:400;
      font-style:normal;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  .bp3-input-group > .bp3-icon,
  .bp3-input-group .bp3-input-action > .bp3-spinner{
    margin:7px; }
  .bp3-input-group .bp3-tag{
    margin:5px; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
    color:#5c7080; }
    .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#a7b6c2; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
      color:#5c7080; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-input-group.bp3-disabled{
    cursor:not-allowed; }
    .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-input-group.bp3-large .bp3-button{
    min-width:30px;
    min-height:30px;
    margin:5px; }
  .bp3-input-group.bp3-large > .bp3-icon,
  .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
    margin:12px; }
  .bp3-input-group.bp3-large .bp3-input{
    height:40px;
    line-height:40px;
    font-size:16px; }
    .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
      padding:0 15px; }
    .bp3-input-group.bp3-large .bp3-input:not(:first-child){
      padding-left:40px; }
    .bp3-input-group.bp3-large .bp3-input:not(:last-child){
      padding-right:40px; }
  .bp3-input-group.bp3-small .bp3-button{
    min-width:20px;
    min-height:20px;
    margin:2px; }
  .bp3-input-group.bp3-small .bp3-tag{
    min-width:20px;
    min-height:20px;
    margin:2px; }
  .bp3-input-group.bp3-small > .bp3-icon,
  .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
    margin:4px; }
  .bp3-input-group.bp3-small .bp3-input{
    height:24px;
    padding-right:8px;
    padding-left:8px;
    line-height:24px;
    font-size:12px; }
    .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
      padding:0 12px; }
    .bp3-input-group.bp3-small .bp3-input:not(:first-child){
      padding-left:24px; }
    .bp3-input-group.bp3-small .bp3-input:not(:last-child){
      padding-right:24px; }
  .bp3-input-group.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-input-group.bp3-round .bp3-button,
  .bp3-input-group.bp3-round .bp3-input,
  .bp3-input-group.bp3-round .bp3-tag{
    border-radius:30px; }
  .bp3-dark .bp3-input-group .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-input-group.bp3-intent-primary .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-primary > .bp3-icon{
    color:#106ba3; }
    .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#48aff0; }
  .bp3-input-group.bp3-intent-success .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-success > .bp3-icon{
    color:#0d8050; }
    .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#3dcc91; }
  .bp3-input-group.bp3-intent-warning .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-warning > .bp3-icon{
    color:#bf7326; }
    .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#ffb366; }
  .bp3-input-group.bp3-intent-danger .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-danger > .bp3-icon{
    color:#c23030; }
    .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#ff7373; }
.bp3-input{
  outline:none;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  height:30px;
  padding:0 10px;
  vertical-align:middle;
  line-height:30px;
  color:#182026;
  font-size:14px;
  font-weight:400;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none; }
  .bp3-input::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input:focus, .bp3-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-input[type="search"], .bp3-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-input:disabled, .bp3-input.bp3-disabled{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(206, 217, 224, 0.5);
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6);
    resize:none; }
  .bp3-input.bp3-large{
    height:40px;
    line-height:40px;
    font-size:16px; }
    .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
      padding:0 15px; }
  .bp3-input.bp3-small{
    height:24px;
    padding-right:8px;
    padding-left:8px;
    line-height:24px;
    font-size:12px; }
    .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
      padding:0 12px; }
  .bp3-input.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-dark .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa; }
    .bp3-dark .bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      color:rgba(167, 182, 194, 0.6); }
  .bp3-input.bp3-intent-primary{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-success{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-warning{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-danger{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input::-ms-clear{
    display:none; }
textarea.bp3-input{
  max-width:100%;
  padding:10px; }
  textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
    height:auto;
    line-height:inherit; }
  textarea.bp3-input.bp3-small{
    padding:8px; }
  .bp3-dark textarea.bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa; }
    .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      color:rgba(167, 182, 194, 0.6); }
label.bp3-label{
  display:block;
  margin-top:0;
  margin-bottom:15px; }
  label.bp3-label .bp3-html-select,
  label.bp3-label .bp3-input,
  label.bp3-label .bp3-select,
  label.bp3-label .bp3-slider,
  label.bp3-label .bp3-popover-wrapper{
    display:block;
    margin-top:5px;
    text-transform:none; }
  label.bp3-label .bp3-button-group{
    margin-top:5px; }
  label.bp3-label .bp3-select select,
  label.bp3-label .bp3-html-select select{
    width:100%;
    vertical-align:top;
    font-weight:400; }
  label.bp3-label.bp3-disabled,
  label.bp3-label.bp3-disabled .bp3-text-muted{
    color:rgba(92, 112, 128, 0.6); }
  label.bp3-label.bp3-inline{
    line-height:30px; }
    label.bp3-label.bp3-inline .bp3-html-select,
    label.bp3-label.bp3-inline .bp3-input,
    label.bp3-label.bp3-inline .bp3-input-group,
    label.bp3-label.bp3-inline .bp3-select,
    label.bp3-label.bp3-inline .bp3-popover-wrapper{
      display:inline-block;
      margin:0 0 0 5px;
      vertical-align:top; }
    label.bp3-label.bp3-inline .bp3-button-group{
      margin:0 0 0 5px; }
    label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
      margin-left:0; }
    label.bp3-label.bp3-inline.bp3-large{
      line-height:40px; }
  label.bp3-label:not(.bp3-inline) .bp3-popover-target{
    display:block; }
  .bp3-dark label.bp3-label{
    color:#f5f8fa; }
    .bp3-dark label.bp3-label.bp3-disabled,
    .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(167, 182, 194, 0.6); }
.bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
  -webkit-box-flex:1;
      -ms-flex:1 1 14px;
          flex:1 1 14px;
  width:30px;
  min-height:0;
  padding:0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
    border-radius:0 3px 0 0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
    border-radius:0 0 3px 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
  border-radius:3px 0 0 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
  border-radius:0 0 0 3px; }

.bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
  width:40px; }

form{
  display:block; }
.bp3-html-select select,
.bp3-select select{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  padding:5px 10px;
  vertical-align:middle;
  text-align:left;
  font-size:14px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  color:#182026;
  border-radius:3px;
  width:100%;
  height:30px;
  padding:0 25px 0 10px;
  -moz-appearance:none;
  -webkit-appearance:none; }
  .bp3-html-select select > *, .bp3-select select > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-html-select select::before,
  .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
    margin-right:7px; }
  .bp3-html-select select:empty::before,
  .bp3-select select:empty::before,
  .bp3-html-select select > :last-child,
  .bp3-select select > :last-child{
    margin-right:0; }
  .bp3-html-select select:hover,
  .bp3-select select:hover{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5; }
  .bp3-html-select select:active,
  .bp3-select select:active, .bp3-html-select select.bp3-active,
  .bp3-select select.bp3-active{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none; }
  .bp3-html-select select:disabled,
  .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
  .bp3-select select.bp3-disabled{
    outline:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
    .bp3-html-select select:disabled.bp3-active,
    .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
    .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
    .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
    .bp3-select select.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }

.bp3-html-select.bp3-minimal select,
.bp3-select.bp3-minimal select{
  -webkit-box-shadow:none;
          box-shadow:none;
  background:none; }
  .bp3-html-select.bp3-minimal select:hover,
  .bp3-select.bp3-minimal select:hover{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(167, 182, 194, 0.3);
    text-decoration:none;
    color:#182026; }
  .bp3-html-select.bp3-minimal select:active,
  .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
  .bp3-select.bp3-minimal select.bp3-active{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(115, 134, 148, 0.3);
    color:#182026; }
  .bp3-html-select.bp3-minimal select:disabled,
  .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
  .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
  .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
  .bp3-select.bp3-minimal select.bp3-disabled:hover{
    background:none;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
    .bp3-html-select.bp3-minimal select:disabled.bp3-active,
    .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
      background:rgba(115, 134, 148, 0.3); }
  .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
  .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none;
    color:inherit; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
      background:rgba(138, 155, 168, 0.15); }
    .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:rgba(138, 155, 168, 0.3);
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
    .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
      background:none;
      cursor:not-allowed;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
        background:rgba(138, 155, 168, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-primary,
  .bp3-select.bp3-minimal select.bp3-intent-primary{
    color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
      background:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:rgba(19, 124, 189, 0.3);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
      background:none;
      color:rgba(16, 107, 163, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
        background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
      stroke:#106ba3; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.2);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(72, 175, 240, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-success,
  .bp3-select.bp3-minimal select.bp3-intent-success{
    color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover{
      background:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:rgba(15, 153, 96, 0.3);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
      background:none;
      color:rgba(13, 128, 80, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
        background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
      stroke:#0d8050; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.2);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(61, 204, 145, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-warning,
  .bp3-select.bp3-minimal select.bp3-intent-warning{
    color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
      background:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:rgba(217, 130, 43, 0.3);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
      background:none;
      color:rgba(191, 115, 38, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
        background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
      stroke:#bf7326; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.2);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(255, 179, 102, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-danger,
  .bp3-select.bp3-minimal select.bp3-intent-danger{
    color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
      background:rgba(219, 55, 55, 0.15);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:rgba(219, 55, 55, 0.3);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
      background:none;
      color:rgba(194, 48, 48, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
        background:rgba(219, 55, 55, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
      stroke:#c23030; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.2);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(255, 115, 115, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }

.bp3-html-select.bp3-large select,
.bp3-select.bp3-large select{
  height:40px;
  padding-right:35px;
  font-size:16px; }

.bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
  background-color:#394b59;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#30404d; }
  .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#202b33;
    background-image:none; }
  .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
    -webkit-box-shadow:none;
            box-shadow:none;
    background-color:rgba(57, 75, 89, 0.5);
    background-image:none;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
      background:rgba(57, 75, 89, 0.7); }
  .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
    background:rgba(16, 22, 26, 0.5);
    stroke:#8a9ba8; }

.bp3-html-select select:disabled,
.bp3-select select:disabled{
  -webkit-box-shadow:none;
          box-shadow:none;
  background-color:rgba(206, 217, 224, 0.5);
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-html-select .bp3-icon,
.bp3-select .bp3-icon, .bp3-select::after{
  position:absolute;
  top:7px;
  right:7px;
  color:#5c7080;
  pointer-events:none; }
  .bp3-html-select .bp3-disabled.bp3-icon,
  .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
    color:rgba(92, 112, 128, 0.6); }
.bp3-html-select,
.bp3-select{
  display:inline-block;
  position:relative;
  vertical-align:middle;
  letter-spacing:normal; }
  .bp3-html-select select::-ms-expand,
  .bp3-select select::-ms-expand{
    display:none; }
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon{
    color:#5c7080; }
    .bp3-html-select .bp3-icon:hover,
    .bp3-select .bp3-icon:hover{
      color:#182026; }
    .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
    .bp3-select .bp3-icon{
      color:#a7b6c2; }
      .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
      .bp3-select .bp3-icon:hover{
        color:#f5f8fa; }
  .bp3-html-select.bp3-large::after,
  .bp3-html-select.bp3-large .bp3-icon,
  .bp3-select.bp3-large::after,
  .bp3-select.bp3-large .bp3-icon{
    top:12px;
    right:12px; }
  .bp3-html-select.bp3-fill,
  .bp3-html-select.bp3-fill select,
  .bp3-select.bp3-fill,
  .bp3-select.bp3-fill select{
    width:100%; }
  .bp3-dark .bp3-html-select option, .bp3-dark
  .bp3-select option{
    background-color:#30404d;
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select::after, .bp3-dark
  .bp3-select::after{
    color:#a7b6c2; }

.bp3-select::after{
  line-height:1;
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-weight:400;
  font-style:normal;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  content:""; }
.bp3-running-text table, table.bp3-html-table{
  border-spacing:0;
  font-size:14px; }
  .bp3-running-text table th, table.bp3-html-table th,
  .bp3-running-text table td,
  table.bp3-html-table td{
    padding:11px;
    vertical-align:top;
    text-align:left; }
  .bp3-running-text table th, table.bp3-html-table th{
    color:#182026;
    font-weight:600; }
  
  .bp3-running-text table td,
  table.bp3-html-table td{
    color:#182026; }
  .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
  .bp3-running-text table tbody tr:first-child td,
  table.bp3-html-table tbody tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }

table.bp3-html-table.bp3-html-table-condensed th,
table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
table.bp3-html-table.bp3-small td{
  padding-top:6px;
  padding-bottom:6px; }

table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(191, 204, 214, 0.15); }

table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered tbody tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:none;
          box-shadow:none; }
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(191, 204, 214, 0.3);
  cursor:pointer; }

table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(191, 204, 214, 0.4); }

.bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(92, 112, 128, 0.15); }

.bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }

.bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }

.bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
    -webkit-box-shadow:none;
            box-shadow:none; }

.bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(92, 112, 128, 0.3);
  cursor:pointer; }

.bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(92, 112, 128, 0.4); }

.bp3-key-combo{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center; }
  .bp3-key-combo > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-key-combo > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-key-combo::before,
  .bp3-key-combo > *{
    margin-right:5px; }
  .bp3-key-combo:empty::before,
  .bp3-key-combo > :last-child{
    margin-right:0; }

.bp3-hotkey-dialog{
  top:40px;
  padding-bottom:0; }
  .bp3-hotkey-dialog .bp3-dialog-body{
    margin:0;
    padding:0; }
  .bp3-hotkey-dialog .bp3-hotkey-label{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1; }

.bp3-hotkey-column{
  margin:auto;
  max-height:80vh;
  overflow-y:auto;
  padding:30px; }
  .bp3-hotkey-column .bp3-heading{
    margin-bottom:20px; }
    .bp3-hotkey-column .bp3-heading:not(:first-child){
      margin-top:40px; }

.bp3-hotkey{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:justify;
      -ms-flex-pack:justify;
          justify-content:space-between;
  margin-right:0;
  margin-left:0; }
  .bp3-hotkey:not(:last-child){
    margin-bottom:10px; }
.bp3-icon{
  display:inline-block;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  vertical-align:text-bottom; }
  .bp3-icon:not(:empty)::before{
    content:"" !important;
    content:unset !important; }
  .bp3-icon > svg{
    display:block; }
    .bp3-icon > svg:not([fill]){
      fill:currentColor; }

.bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
  color:#106ba3; }
  .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
    color:#48aff0; }

.bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
  color:#0d8050; }
  .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
    color:#3dcc91; }

.bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
  color:#bf7326; }
  .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
    color:#ffb366; }

.bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
  color:#c23030; }
  .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
    color:#ff7373; }

span.bp3-icon-standard{
  line-height:1;
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-weight:400;
  font-style:normal;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon-large{
  line-height:1;
  font-family:"Icons20", sans-serif;
  font-size:20px;
  font-weight:400;
  font-style:normal;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon:empty{
  line-height:1;
  font-family:"Icons20";
  font-size:inherit;
  font-weight:400;
  font-style:normal; }
  span.bp3-icon:empty::before{
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased; }

.bp3-icon-add::before{
  content:""; }

.bp3-icon-add-column-left::before{
  content:""; }

.bp3-icon-add-column-right::before{
  content:""; }

.bp3-icon-add-row-bottom::before{
  content:""; }

.bp3-icon-add-row-top::before{
  content:""; }

.bp3-icon-add-to-artifact::before{
  content:""; }

.bp3-icon-add-to-folder::before{
  content:""; }

.bp3-icon-airplane::before{
  content:""; }

.bp3-icon-align-center::before{
  content:""; }

.bp3-icon-align-justify::before{
  content:""; }

.bp3-icon-align-left::before{
  content:""; }

.bp3-icon-align-right::before{
  content:""; }

.bp3-icon-alignment-bottom::before{
  content:""; }

.bp3-icon-alignment-horizontal-center::before{
  content:""; }

.bp3-icon-alignment-left::before{
  content:""; }

.bp3-icon-alignment-right::before{
  content:""; }

.bp3-icon-alignment-top::before{
  content:""; }

.bp3-icon-alignment-vertical-center::before{
  content:""; }

.bp3-icon-annotation::before{
  content:""; }

.bp3-icon-application::before{
  content:""; }

.bp3-icon-applications::before{
  content:""; }

.bp3-icon-archive::before{
  content:""; }

.bp3-icon-arrow-bottom-left::before{
  content:"↙"; }

.bp3-icon-arrow-bottom-right::before{
  content:"↘"; }

.bp3-icon-arrow-down::before{
  content:"↓"; }

.bp3-icon-arrow-left::before{
  content:"←"; }

.bp3-icon-arrow-right::before{
  content:"→"; }

.bp3-icon-arrow-top-left::before{
  content:"↖"; }

.bp3-icon-arrow-top-right::before{
  content:"↗"; }

.bp3-icon-arrow-up::before{
  content:"↑"; }

.bp3-icon-arrows-horizontal::before{
  content:"↔"; }

.bp3-icon-arrows-vertical::before{
  content:"↕"; }

.bp3-icon-asterisk::before{
  content:"*"; }

.bp3-icon-automatic-updates::before{
  content:""; }

.bp3-icon-badge::before{
  content:""; }

.bp3-icon-ban-circle::before{
  content:""; }

.bp3-icon-bank-account::before{
  content:""; }

.bp3-icon-barcode::before{
  content:""; }

.bp3-icon-blank::before{
  content:""; }

.bp3-icon-blocked-person::before{
  content:""; }

.bp3-icon-bold::before{
  content:""; }

.bp3-icon-book::before{
  content:""; }

.bp3-icon-bookmark::before{
  content:""; }

.bp3-icon-box::before{
  content:""; }

.bp3-icon-briefcase::before{
  content:""; }

.bp3-icon-bring-data::before{
  content:""; }

.bp3-icon-build::before{
  content:""; }

.bp3-icon-calculator::before{
  content:""; }

.bp3-icon-calendar::before{
  content:""; }

.bp3-icon-camera::before{
  content:""; }

.bp3-icon-caret-down::before{
  content:"⌄"; }

.bp3-icon-caret-left::before{
  content:"〈"; }

.bp3-icon-caret-right::before{
  content:"〉"; }

.bp3-icon-caret-up::before{
  content:"⌃"; }

.bp3-icon-cell-tower::before{
  content:""; }

.bp3-icon-changes::before{
  content:""; }

.bp3-icon-chart::before{
  content:""; }

.bp3-icon-chat::before{
  content:""; }

.bp3-icon-chevron-backward::before{
  content:""; }

.bp3-icon-chevron-down::before{
  content:""; }

.bp3-icon-chevron-forward::before{
  content:""; }

.bp3-icon-chevron-left::before{
  content:""; }

.bp3-icon-chevron-right::before{
  content:""; }

.bp3-icon-chevron-up::before{
  content:""; }

.bp3-icon-circle::before{
  content:""; }

.bp3-icon-circle-arrow-down::before{
  content:""; }

.bp3-icon-circle-arrow-left::before{
  content:""; }

.bp3-icon-circle-arrow-right::before{
  content:""; }

.bp3-icon-circle-arrow-up::before{
  content:""; }

.bp3-icon-citation::before{
  content:""; }

.bp3-icon-clean::before{
  content:""; }

.bp3-icon-clipboard::before{
  content:""; }

.bp3-icon-cloud::before{
  content:"☁"; }

.bp3-icon-cloud-download::before{
  content:""; }

.bp3-icon-cloud-upload::before{
  content:""; }

.bp3-icon-code::before{
  content:""; }

.bp3-icon-code-block::before{
  content:""; }

.bp3-icon-cog::before{
  content:""; }

.bp3-icon-collapse-all::before{
  content:""; }

.bp3-icon-column-layout::before{
  content:""; }

.bp3-icon-comment::before{
  content:""; }

.bp3-icon-comparison::before{
  content:""; }

.bp3-icon-compass::before{
  content:""; }

.bp3-icon-compressed::before{
  content:""; }

.bp3-icon-confirm::before{
  content:""; }

.bp3-icon-console::before{
  content:""; }

.bp3-icon-contrast::before{
  content:""; }

.bp3-icon-control::before{
  content:""; }

.bp3-icon-credit-card::before{
  content:""; }

.bp3-icon-cross::before{
  content:"✗"; }

.bp3-icon-crown::before{
  content:""; }

.bp3-icon-cube::before{
  content:""; }

.bp3-icon-cube-add::before{
  content:""; }

.bp3-icon-cube-remove::before{
  content:""; }

.bp3-icon-curved-range-chart::before{
  content:""; }

.bp3-icon-cut::before{
  content:""; }

.bp3-icon-dashboard::before{
  content:""; }

.bp3-icon-data-lineage::before{
  content:""; }

.bp3-icon-database::before{
  content:""; }

.bp3-icon-delete::before{
  content:""; }

.bp3-icon-delta::before{
  content:"Δ"; }

.bp3-icon-derive-column::before{
  content:""; }

.bp3-icon-desktop::before{
  content:""; }

.bp3-icon-diagram-tree::before{
  content:""; }

.bp3-icon-direction-left::before{
  content:""; }

.bp3-icon-direction-right::before{
  content:""; }

.bp3-icon-disable::before{
  content:""; }

.bp3-icon-document::before{
  content:""; }

.bp3-icon-document-open::before{
  content:""; }

.bp3-icon-document-share::before{
  content:""; }

.bp3-icon-dollar::before{
  content:"$"; }

.bp3-icon-dot::before{
  content:"•"; }

.bp3-icon-double-caret-horizontal::before{
  content:""; }

.bp3-icon-double-caret-vertical::before{
  content:""; }

.bp3-icon-double-chevron-down::before{
  content:""; }

.bp3-icon-double-chevron-left::before{
  content:""; }

.bp3-icon-double-chevron-right::before{
  content:""; }

.bp3-icon-double-chevron-up::before{
  content:""; }

.bp3-icon-doughnut-chart::before{
  content:""; }

.bp3-icon-download::before{
  content:""; }

.bp3-icon-drag-handle-horizontal::before{
  content:""; }

.bp3-icon-drag-handle-vertical::before{
  content:""; }

.bp3-icon-draw::before{
  content:""; }

.bp3-icon-drive-time::before{
  content:""; }

.bp3-icon-duplicate::before{
  content:""; }

.bp3-icon-edit::before{
  content:"✎"; }

.bp3-icon-eject::before{
  content:"⏏"; }

.bp3-icon-endorsed::before{
  content:""; }

.bp3-icon-envelope::before{
  content:"✉"; }

.bp3-icon-equals::before{
  content:""; }

.bp3-icon-eraser::before{
  content:""; }

.bp3-icon-error::before{
  content:""; }

.bp3-icon-euro::before{
  content:"€"; }

.bp3-icon-exchange::before{
  content:""; }

.bp3-icon-exclude-row::before{
  content:""; }

.bp3-icon-expand-all::before{
  content:""; }

.bp3-icon-export::before{
  content:""; }

.bp3-icon-eye-off::before{
  content:""; }

.bp3-icon-eye-on::before{
  content:""; }

.bp3-icon-eye-open::before{
  content:""; }

.bp3-icon-fast-backward::before{
  content:""; }

.bp3-icon-fast-forward::before{
  content:""; }

.bp3-icon-feed::before{
  content:""; }

.bp3-icon-feed-subscribed::before{
  content:""; }

.bp3-icon-film::before{
  content:""; }

.bp3-icon-filter::before{
  content:""; }

.bp3-icon-filter-keep::before{
  content:""; }

.bp3-icon-filter-list::before{
  content:""; }

.bp3-icon-filter-open::before{
  content:""; }

.bp3-icon-filter-remove::before{
  content:""; }

.bp3-icon-flag::before{
  content:"⚑"; }

.bp3-icon-flame::before{
  content:""; }

.bp3-icon-flash::before{
  content:""; }

.bp3-icon-floppy-disk::before{
  content:""; }

.bp3-icon-flow-branch::before{
  content:""; }

.bp3-icon-flow-end::before{
  content:""; }

.bp3-icon-flow-linear::before{
  content:""; }

.bp3-icon-flow-review::before{
  content:""; }

.bp3-icon-flow-review-branch::before{
  content:""; }

.bp3-icon-flows::before{
  content:""; }

.bp3-icon-folder-close::before{
  content:""; }

.bp3-icon-folder-new::before{
  content:""; }

.bp3-icon-folder-open::before{
  content:""; }

.bp3-icon-folder-shared::before{
  content:""; }

.bp3-icon-folder-shared-open::before{
  content:""; }

.bp3-icon-follower::before{
  content:""; }

.bp3-icon-following::before{
  content:""; }

.bp3-icon-font::before{
  content:""; }

.bp3-icon-fork::before{
  content:""; }

.bp3-icon-form::before{
  content:""; }

.bp3-icon-full-circle::before{
  content:""; }

.bp3-icon-full-stacked-chart::before{
  content:""; }

.bp3-icon-fullscreen::before{
  content:""; }

.bp3-icon-function::before{
  content:""; }

.bp3-icon-gantt-chart::before{
  content:""; }

.bp3-icon-geolocation::before{
  content:""; }

.bp3-icon-geosearch::before{
  content:""; }

.bp3-icon-git-branch::before{
  content:""; }

.bp3-icon-git-commit::before{
  content:""; }

.bp3-icon-git-merge::before{
  content:""; }

.bp3-icon-git-new-branch::before{
  content:""; }

.bp3-icon-git-pull::before{
  content:""; }

.bp3-icon-git-push::before{
  content:""; }

.bp3-icon-git-repo::before{
  content:""; }

.bp3-icon-glass::before{
  content:""; }

.bp3-icon-globe::before{
  content:""; }

.bp3-icon-globe-network::before{
  content:""; }

.bp3-icon-graph::before{
  content:""; }

.bp3-icon-graph-remove::before{
  content:""; }

.bp3-icon-greater-than::before{
  content:""; }

.bp3-icon-greater-than-or-equal-to::before{
  content:""; }

.bp3-icon-grid::before{
  content:""; }

.bp3-icon-grid-view::before{
  content:""; }

.bp3-icon-group-objects::before{
  content:""; }

.bp3-icon-grouped-bar-chart::before{
  content:""; }

.bp3-icon-hand::before{
  content:""; }

.bp3-icon-hand-down::before{
  content:""; }

.bp3-icon-hand-left::before{
  content:""; }

.bp3-icon-hand-right::before{
  content:""; }

.bp3-icon-hand-up::before{
  content:""; }

.bp3-icon-header::before{
  content:""; }

.bp3-icon-header-one::before{
  content:""; }

.bp3-icon-header-two::before{
  content:""; }

.bp3-icon-headset::before{
  content:""; }

.bp3-icon-heart::before{
  content:"♥"; }

.bp3-icon-heart-broken::before{
  content:""; }

.bp3-icon-heat-grid::before{
  content:""; }

.bp3-icon-heatmap::before{
  content:""; }

.bp3-icon-help::before{
  content:"?"; }

.bp3-icon-helper-management::before{
  content:""; }

.bp3-icon-highlight::before{
  content:""; }

.bp3-icon-history::before{
  content:""; }

.bp3-icon-home::before{
  content:"⌂"; }

.bp3-icon-horizontal-bar-chart::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-asc::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-desc::before{
  content:""; }

.bp3-icon-horizontal-distribution::before{
  content:""; }

.bp3-icon-id-number::before{
  content:""; }

.bp3-icon-image-rotate-left::before{
  content:""; }

.bp3-icon-image-rotate-right::before{
  content:""; }

.bp3-icon-import::before{
  content:""; }

.bp3-icon-inbox::before{
  content:""; }

.bp3-icon-inbox-filtered::before{
  content:""; }

.bp3-icon-inbox-geo::before{
  content:""; }

.bp3-icon-inbox-search::before{
  content:""; }

.bp3-icon-inbox-update::before{
  content:""; }

.bp3-icon-info-sign::before{
  content:"ℹ"; }

.bp3-icon-inheritance::before{
  content:""; }

.bp3-icon-inner-join::before{
  content:""; }

.bp3-icon-insert::before{
  content:""; }

.bp3-icon-intersection::before{
  content:""; }

.bp3-icon-ip-address::before{
  content:""; }

.bp3-icon-issue::before{
  content:""; }

.bp3-icon-issue-closed::before{
  content:""; }

.bp3-icon-issue-new::before{
  content:""; }

.bp3-icon-italic::before{
  content:""; }

.bp3-icon-join-table::before{
  content:""; }

.bp3-icon-key::before{
  content:""; }

.bp3-icon-key-backspace::before{
  content:""; }

.bp3-icon-key-command::before{
  content:""; }

.bp3-icon-key-control::before{
  content:""; }

.bp3-icon-key-delete::before{
  content:""; }

.bp3-icon-key-enter::before{
  content:""; }

.bp3-icon-key-escape::before{
  content:""; }

.bp3-icon-key-option::before{
  content:""; }

.bp3-icon-key-shift::before{
  content:""; }

.bp3-icon-key-tab::before{
  content:""; }

.bp3-icon-known-vehicle::before{
  content:""; }

.bp3-icon-label::before{
  content:""; }

.bp3-icon-layer::before{
  content:""; }

.bp3-icon-layers::before{
  content:""; }

.bp3-icon-layout::before{
  content:""; }

.bp3-icon-layout-auto::before{
  content:""; }

.bp3-icon-layout-balloon::before{
  content:""; }

.bp3-icon-layout-circle::before{
  content:""; }

.bp3-icon-layout-grid::before{
  content:""; }

.bp3-icon-layout-group-by::before{
  content:""; }

.bp3-icon-layout-hierarchy::before{
  content:""; }

.bp3-icon-layout-linear::before{
  content:""; }

.bp3-icon-layout-skew-grid::before{
  content:""; }

.bp3-icon-layout-sorted-clusters::before{
  content:""; }

.bp3-icon-learning::before{
  content:""; }

.bp3-icon-left-join::before{
  content:""; }

.bp3-icon-less-than::before{
  content:""; }

.bp3-icon-less-than-or-equal-to::before{
  content:""; }

.bp3-icon-lifesaver::before{
  content:""; }

.bp3-icon-lightbulb::before{
  content:""; }

.bp3-icon-link::before{
  content:""; }

.bp3-icon-list::before{
  content:"☰"; }

.bp3-icon-list-columns::before{
  content:""; }

.bp3-icon-list-detail-view::before{
  content:""; }

.bp3-icon-locate::before{
  content:""; }

.bp3-icon-lock::before{
  content:""; }

.bp3-icon-log-in::before{
  content:""; }

.bp3-icon-log-out::before{
  content:""; }

.bp3-icon-manual::before{
  content:""; }

.bp3-icon-manually-entered-data::before{
  content:""; }

.bp3-icon-map::before{
  content:""; }

.bp3-icon-map-create::before{
  content:""; }

.bp3-icon-map-marker::before{
  content:""; }

.bp3-icon-maximize::before{
  content:""; }

.bp3-icon-media::before{
  content:""; }

.bp3-icon-menu::before{
  content:""; }

.bp3-icon-menu-closed::before{
  content:""; }

.bp3-icon-menu-open::before{
  content:""; }

.bp3-icon-merge-columns::before{
  content:""; }

.bp3-icon-merge-links::before{
  content:""; }

.bp3-icon-minimize::before{
  content:""; }

.bp3-icon-minus::before{
  content:"−"; }

.bp3-icon-mobile-phone::before{
  content:""; }

.bp3-icon-mobile-video::before{
  content:""; }

.bp3-icon-moon::before{
  content:""; }

.bp3-icon-more::before{
  content:""; }

.bp3-icon-mountain::before{
  content:""; }

.bp3-icon-move::before{
  content:""; }

.bp3-icon-mugshot::before{
  content:""; }

.bp3-icon-multi-select::before{
  content:""; }

.bp3-icon-music::before{
  content:""; }

.bp3-icon-new-drawing::before{
  content:""; }

.bp3-icon-new-grid-item::before{
  content:""; }

.bp3-icon-new-layer::before{
  content:""; }

.bp3-icon-new-layers::before{
  content:""; }

.bp3-icon-new-link::before{
  content:""; }

.bp3-icon-new-object::before{
  content:""; }

.bp3-icon-new-person::before{
  content:""; }

.bp3-icon-new-prescription::before{
  content:""; }

.bp3-icon-new-text-box::before{
  content:""; }

.bp3-icon-ninja::before{
  content:""; }

.bp3-icon-not-equal-to::before{
  content:""; }

.bp3-icon-notifications::before{
  content:""; }

.bp3-icon-notifications-updated::before{
  content:""; }

.bp3-icon-numbered-list::before{
  content:""; }

.bp3-icon-numerical::before{
  content:""; }

.bp3-icon-office::before{
  content:""; }

.bp3-icon-offline::before{
  content:""; }

.bp3-icon-oil-field::before{
  content:""; }

.bp3-icon-one-column::before{
  content:""; }

.bp3-icon-outdated::before{
  content:""; }

.bp3-icon-page-layout::before{
  content:""; }

.bp3-icon-panel-stats::before{
  content:""; }

.bp3-icon-panel-table::before{
  content:""; }

.bp3-icon-paperclip::before{
  content:""; }

.bp3-icon-paragraph::before{
  content:""; }

.bp3-icon-path::before{
  content:""; }

.bp3-icon-path-search::before{
  content:""; }

.bp3-icon-pause::before{
  content:""; }

.bp3-icon-people::before{
  content:""; }

.bp3-icon-percentage::before{
  content:""; }

.bp3-icon-person::before{
  content:""; }

.bp3-icon-phone::before{
  content:"☎"; }

.bp3-icon-pie-chart::before{
  content:""; }

.bp3-icon-pin::before{
  content:""; }

.bp3-icon-pivot::before{
  content:""; }

.bp3-icon-pivot-table::before{
  content:""; }

.bp3-icon-play::before{
  content:""; }

.bp3-icon-plus::before{
  content:"+"; }

.bp3-icon-polygon-filter::before{
  content:""; }

.bp3-icon-power::before{
  content:""; }

.bp3-icon-predictive-analysis::before{
  content:""; }

.bp3-icon-prescription::before{
  content:""; }

.bp3-icon-presentation::before{
  content:""; }

.bp3-icon-print::before{
  content:"⎙"; }

.bp3-icon-projects::before{
  content:""; }

.bp3-icon-properties::before{
  content:""; }

.bp3-icon-property::before{
  content:""; }

.bp3-icon-publish-function::before{
  content:""; }

.bp3-icon-pulse::before{
  content:""; }

.bp3-icon-random::before{
  content:""; }

.bp3-icon-record::before{
  content:""; }

.bp3-icon-redo::before{
  content:""; }

.bp3-icon-refresh::before{
  content:""; }

.bp3-icon-regression-chart::before{
  content:""; }

.bp3-icon-remove::before{
  content:""; }

.bp3-icon-remove-column::before{
  content:""; }

.bp3-icon-remove-column-left::before{
  content:""; }

.bp3-icon-remove-column-right::before{
  content:""; }

.bp3-icon-remove-row-bottom::before{
  content:""; }

.bp3-icon-remove-row-top::before{
  content:""; }

.bp3-icon-repeat::before{
  content:""; }

.bp3-icon-reset::before{
  content:""; }

.bp3-icon-resolve::before{
  content:""; }

.bp3-icon-rig::before{
  content:""; }

.bp3-icon-right-join::before{
  content:""; }

.bp3-icon-ring::before{
  content:""; }

.bp3-icon-rotate-document::before{
  content:""; }

.bp3-icon-rotate-page::before{
  content:""; }

.bp3-icon-satellite::before{
  content:""; }

.bp3-icon-saved::before{
  content:""; }

.bp3-icon-scatter-plot::before{
  content:""; }

.bp3-icon-search::before{
  content:""; }

.bp3-icon-search-around::before{
  content:""; }

.bp3-icon-search-template::before{
  content:""; }

.bp3-icon-search-text::before{
  content:""; }

.bp3-icon-segmented-control::before{
  content:""; }

.bp3-icon-select::before{
  content:""; }

.bp3-icon-selection::before{
  content:"⦿"; }

.bp3-icon-send-to::before{
  content:""; }

.bp3-icon-send-to-graph::before{
  content:""; }

.bp3-icon-send-to-map::before{
  content:""; }

.bp3-icon-series-add::before{
  content:""; }

.bp3-icon-series-configuration::before{
  content:""; }

.bp3-icon-series-derived::before{
  content:""; }

.bp3-icon-series-filtered::before{
  content:""; }

.bp3-icon-series-search::before{
  content:""; }

.bp3-icon-settings::before{
  content:""; }

.bp3-icon-share::before{
  content:""; }

.bp3-icon-shield::before{
  content:""; }

.bp3-icon-shop::before{
  content:""; }

.bp3-icon-shopping-cart::before{
  content:""; }

.bp3-icon-signal-search::before{
  content:""; }

.bp3-icon-sim-card::before{
  content:""; }

.bp3-icon-slash::before{
  content:""; }

.bp3-icon-small-cross::before{
  content:""; }

.bp3-icon-small-minus::before{
  content:""; }

.bp3-icon-small-plus::before{
  content:""; }

.bp3-icon-small-tick::before{
  content:""; }

.bp3-icon-snowflake::before{
  content:""; }

.bp3-icon-social-media::before{
  content:""; }

.bp3-icon-sort::before{
  content:""; }

.bp3-icon-sort-alphabetical::before{
  content:""; }

.bp3-icon-sort-alphabetical-desc::before{
  content:""; }

.bp3-icon-sort-asc::before{
  content:""; }

.bp3-icon-sort-desc::before{
  content:""; }

.bp3-icon-sort-numerical::before{
  content:""; }

.bp3-icon-sort-numerical-desc::before{
  content:""; }

.bp3-icon-split-columns::before{
  content:""; }

.bp3-icon-square::before{
  content:""; }

.bp3-icon-stacked-chart::before{
  content:""; }

.bp3-icon-star::before{
  content:"★"; }

.bp3-icon-star-empty::before{
  content:"☆"; }

.bp3-icon-step-backward::before{
  content:""; }

.bp3-icon-step-chart::before{
  content:""; }

.bp3-icon-step-forward::before{
  content:""; }

.bp3-icon-stop::before{
  content:""; }

.bp3-icon-stopwatch::before{
  content:""; }

.bp3-icon-strikethrough::before{
  content:""; }

.bp3-icon-style::before{
  content:""; }

.bp3-icon-swap-horizontal::before{
  content:""; }

.bp3-icon-swap-vertical::before{
  content:""; }

.bp3-icon-symbol-circle::before{
  content:""; }

.bp3-icon-symbol-cross::before{
  content:""; }

.bp3-icon-symbol-diamond::before{
  content:""; }

.bp3-icon-symbol-square::before{
  content:""; }

.bp3-icon-symbol-triangle-down::before{
  content:""; }

.bp3-icon-symbol-triangle-up::before{
  content:""; }

.bp3-icon-tag::before{
  content:""; }

.bp3-icon-take-action::before{
  content:""; }

.bp3-icon-taxi::before{
  content:""; }

.bp3-icon-text-highlight::before{
  content:""; }

.bp3-icon-th::before{
  content:""; }

.bp3-icon-th-derived::before{
  content:""; }

.bp3-icon-th-disconnect::before{
  content:""; }

.bp3-icon-th-filtered::before{
  content:""; }

.bp3-icon-th-list::before{
  content:""; }

.bp3-icon-thumbs-down::before{
  content:""; }

.bp3-icon-thumbs-up::before{
  content:""; }

.bp3-icon-tick::before{
  content:"✓"; }

.bp3-icon-tick-circle::before{
  content:""; }

.bp3-icon-time::before{
  content:"⏲"; }

.bp3-icon-timeline-area-chart::before{
  content:""; }

.bp3-icon-timeline-bar-chart::before{
  content:""; }

.bp3-icon-timeline-events::before{
  content:""; }

.bp3-icon-timeline-line-chart::before{
  content:""; }

.bp3-icon-tint::before{
  content:""; }

.bp3-icon-torch::before{
  content:""; }

.bp3-icon-tractor::before{
  content:""; }

.bp3-icon-train::before{
  content:""; }

.bp3-icon-translate::before{
  content:""; }

.bp3-icon-trash::before{
  content:""; }

.bp3-icon-tree::before{
  content:""; }

.bp3-icon-trending-down::before{
  content:""; }

.bp3-icon-trending-up::before{
  content:""; }

.bp3-icon-truck::before{
  content:""; }

.bp3-icon-two-columns::before{
  content:""; }

.bp3-icon-unarchive::before{
  content:""; }

.bp3-icon-underline::before{
  content:"⎁"; }

.bp3-icon-undo::before{
  content:"⎌"; }

.bp3-icon-ungroup-objects::before{
  content:""; }

.bp3-icon-unknown-vehicle::before{
  content:""; }

.bp3-icon-unlock::before{
  content:""; }

.bp3-icon-unpin::before{
  content:""; }

.bp3-icon-unresolve::before{
  content:""; }

.bp3-icon-updated::before{
  content:""; }

.bp3-icon-upload::before{
  content:""; }

.bp3-icon-user::before{
  content:""; }

.bp3-icon-variable::before{
  content:""; }

.bp3-icon-vertical-bar-chart-asc::before{
  content:""; }

.bp3-icon-vertical-bar-chart-desc::before{
  content:""; }

.bp3-icon-vertical-distribution::before{
  content:""; }

.bp3-icon-video::before{
  content:""; }

.bp3-icon-volume-down::before{
  content:""; }

.bp3-icon-volume-off::before{
  content:""; }

.bp3-icon-volume-up::before{
  content:""; }

.bp3-icon-walk::before{
  content:""; }

.bp3-icon-warning-sign::before{
  content:""; }

.bp3-icon-waterfall-chart::before{
  content:""; }

.bp3-icon-widget::before{
  content:""; }

.bp3-icon-widget-button::before{
  content:""; }

.bp3-icon-widget-footer::before{
  content:""; }

.bp3-icon-widget-header::before{
  content:""; }

.bp3-icon-wrench::before{
  content:""; }

.bp3-icon-zoom-in::before{
  content:""; }

.bp3-icon-zoom-out::before{
  content:""; }

.bp3-icon-zoom-to-fit::before{
  content:""; }
.bp3-submenu > .bp3-popover-wrapper{
  display:block; }

.bp3-submenu .bp3-popover-target{
  display:block; }

.bp3-submenu.bp3-popover{
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0 5px; }
  .bp3-submenu.bp3-popover > .bp3-popover-content{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
.bp3-menu{
  margin:0;
  border-radius:3px;
  background:#ffffff;
  min-width:180px;
  padding:5px;
  list-style:none;
  text-align:left;
  color:#182026; }

.bp3-menu-divider{
  display:block;
  margin:5px;
  border-top:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-menu-divider{
    border-top-color:rgba(255, 255, 255, 0.15); }

.bp3-menu-item{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  border-radius:2px;
  padding:5px 7px;
  text-decoration:none;
  line-height:20px;
  color:inherit;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-menu-item > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-menu-item > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-menu-item::before,
  .bp3-menu-item > *{
    margin-right:7px; }
  .bp3-menu-item:empty::before,
  .bp3-menu-item > :last-child{
    margin-right:0; }
  .bp3-menu-item > .bp3-fill{
    word-break:break-word; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    background-color:rgba(167, 182, 194, 0.3);
    cursor:pointer;
    text-decoration:none; }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-menu-item{
    color:inherit; }
    .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(138, 155, 168, 0.15);
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-menu-item.bp3-intent-primary{
    color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
    .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-success{
    color:#0d8050; }
    .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
    .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-warning{
    color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
    .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-danger{
    color:#c23030; }
    .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
    .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item::before{
    line-height:1;
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-weight:400;
    font-style:normal;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    margin-right:7px; }
  .bp3-menu-item::before,
  .bp3-menu-item > .bp3-icon{
    margin-top:2px;
    color:#5c7080; }
  .bp3-menu-item .bp3-menu-item-label{
    color:#5c7080; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    color:inherit; }
  .bp3-menu-item.bp3-active, .bp3-menu-item:active{
    background-color:rgba(115, 134, 148, 0.3); }
  .bp3-menu-item.bp3-disabled{
    outline:none !important;
    background-color:inherit !important;
    cursor:not-allowed !important;
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-menu-item.bp3-disabled::before,
    .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-large .bp3-menu-item{
    padding:9px 7px;
    line-height:22px;
    font-size:16px; }
    .bp3-large .bp3-menu-item .bp3-icon{
      margin-top:3px; }
    .bp3-large .bp3-menu-item::before{
      line-height:1;
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-weight:400;
      font-style:normal;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-top:1px;
      margin-right:10px; }

button.bp3-menu-item{
  border:none;
  background:none;
  width:100%;
  text-align:left; }
.bp3-menu-header{
  display:block;
  margin:5px;
  border-top:1px solid rgba(16, 22, 26, 0.15);
  cursor:default;
  padding-left:2px; }
  .bp3-dark .bp3-menu-header{
    border-top-color:rgba(255, 255, 255, 0.15); }
  .bp3-menu-header:first-of-type{
    border-top:none; }
  .bp3-menu-header > h6{
    color:#182026;
    font-weight:600;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    margin:0;
    padding:10px 7px 0 1px;
    line-height:17px; }
    .bp3-dark .bp3-menu-header > h6{
      color:#f5f8fa; }
  .bp3-menu-header:first-of-type > h6{
    padding-top:0; }
  .bp3-large .bp3-menu-header > h6{
    padding-top:15px;
    padding-bottom:5px;
    font-size:18px; }
  .bp3-large .bp3-menu-header:first-of-type > h6{
    padding-top:0; }

.bp3-dark .bp3-menu{
  background:#30404d;
  color:#f5f8fa; }

.bp3-dark .bp3-menu-item.bp3-intent-primary{
  color:#48aff0; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
    color:#48aff0; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
    background-color:#137cbd; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
    background-color:#106ba3; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item.bp3-intent-success{
  color:#3dcc91; }
  .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
    color:#3dcc91; }
  .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
    background-color:#0f9960; }
  .bp3-dark .bp3-menu-item.bp3-intent-success:active{
    background-color:#0d8050; }
  .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item.bp3-intent-warning{
  color:#ffb366; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
    color:#ffb366; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
    background-color:#d9822b; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
    background-color:#bf7326; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item.bp3-intent-danger{
  color:#ff7373; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
    color:#ff7373; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
    background-color:#db3737; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
    background-color:#c23030; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item::before,
.bp3-dark .bp3-menu-item > .bp3-icon{
  color:#a7b6c2; }

.bp3-dark .bp3-menu-item .bp3-menu-item-label{
  color:#a7b6c2; }

.bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
  background-color:rgba(138, 155, 168, 0.3); }

.bp3-dark .bp3-menu-item.bp3-disabled{
  color:rgba(167, 182, 194, 0.6) !important; }
  .bp3-dark .bp3-menu-item.bp3-disabled::before,
  .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
  .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
    color:rgba(167, 182, 194, 0.6) !important; }

.bp3-dark .bp3-menu-divider,
.bp3-dark .bp3-menu-header{
  border-color:rgba(255, 255, 255, 0.15); }

.bp3-dark .bp3-menu-header > h6{
  color:#f5f8fa; }

.bp3-label .bp3-menu{
  margin-top:5px; }
.bp3-navbar{
  position:relative;
  z-index:10;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  background-color:#ffffff;
  width:100%;
  height:50px;
  padding:0 15px; }
  .bp3-navbar.bp3-dark,
  .bp3-dark .bp3-navbar{
    background-color:#394b59; }
  .bp3-navbar.bp3-dark{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-navbar{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-navbar.bp3-fixed-top{
    position:fixed;
    top:0;
    right:0;
    left:0; }

.bp3-navbar-heading{
  margin-right:15px;
  font-size:16px; }

.bp3-navbar-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  height:50px; }
  .bp3-navbar-group.bp3-align-left{
    float:left; }
  .bp3-navbar-group.bp3-align-right{
    float:right; }

.bp3-navbar-divider{
  margin:0 10px;
  border-left:1px solid rgba(16, 22, 26, 0.15);
  height:20px; }
  .bp3-dark .bp3-navbar-divider{
    border-left-color:rgba(255, 255, 255, 0.15); }
.bp3-non-ideal-state{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:100%;
  height:100%;
  text-align:center; }
  .bp3-non-ideal-state > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-non-ideal-state > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-non-ideal-state::before,
  .bp3-non-ideal-state > *{
    margin-bottom:20px; }
  .bp3-non-ideal-state:empty::before,
  .bp3-non-ideal-state > :last-child{
    margin-bottom:0; }
  .bp3-non-ideal-state > *{
    max-width:400px; }

.bp3-non-ideal-state-visual{
  color:rgba(92, 112, 128, 0.6);
  font-size:60px; }
  .bp3-dark .bp3-non-ideal-state-visual{
    color:rgba(167, 182, 194, 0.6); }

.bp3-overflow-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:nowrap;
      flex-wrap:nowrap;
  min-width:0; }

.bp3-overflow-list-spacer{
  -ms-flex-negative:1;
      flex-shrink:1;
  width:1px; }

body.bp3-overlay-open{
  overflow:hidden; }

.bp3-overlay{
  position:static;
  top:0;
  right:0;
  bottom:0;
  left:0;
  z-index:20; }
  .bp3-overlay:not(.bp3-overlay-open){
    pointer-events:none; }
  .bp3-overlay.bp3-overlay-container{
    position:fixed;
    overflow:hidden; }
    .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-scroll-container{
    position:fixed;
    overflow:auto; }
    .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-inline{
    display:inline;
    overflow:visible; }

.bp3-overlay-content{
  position:fixed;
  z-index:20; }
  .bp3-overlay-inline .bp3-overlay-content,
  .bp3-overlay-scroll-container .bp3-overlay-content{
    position:absolute; }

.bp3-overlay-backdrop{
  position:fixed;
  top:0;
  right:0;
  bottom:0;
  left:0;
  opacity:1;
  z-index:20;
  background-color:rgba(16, 22, 26, 0.7);
  overflow:auto;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
    opacity:0; }
  .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
    opacity:1;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-overlay-backdrop.bp3-overlay-exit{
    opacity:1; }
  .bp3-overlay-backdrop.bp3-overlay-exit-active{
    opacity:0;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-overlay-backdrop:focus{
    outline:none; }
  .bp3-overlay-inline .bp3-overlay-backdrop{
    position:absolute; }
.bp3-panel-stack{
  position:relative;
  overflow:hidden; }

.bp3-panel-stack-header{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  z-index:1;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  height:30px; }
  .bp3-dark .bp3-panel-stack-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack-header > span{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1;
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch; }
  .bp3-panel-stack-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack-view{
  position:absolute;
  top:0;
  right:0;
  bottom:0;
  left:0;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  background-color:#ffffff;
  overflow-y:auto; }
  .bp3-dark .bp3-panel-stack-view{
    background-color:#30404d; }

.bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }

.bp3-panel-stack-push .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-push .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }

.bp3-panel-stack-pop .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-pop .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }
.bp3-popover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1);
  display:inline-block;
  z-index:20;
  border-radius:3px; }
  .bp3-popover .bp3-popover-arrow{
    position:absolute;
    width:30px;
    height:30px; }
    .bp3-popover .bp3-popover-arrow::before{
      margin:5px;
      width:20px;
      height:20px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
    margin-top:-17px;
    margin-bottom:17px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      bottom:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
    margin-left:17px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      left:-11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
    margin-top:17px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      top:-11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
    margin-right:17px;
    margin-left:-17px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      right:-11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
    top:-0.3934px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
    right:-0.3934px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
    left:-0.3934px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
    bottom:-0.3934px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-popover .bp3-popover-content{
    background:#ffffff;
    color:inherit; }
  .bp3-popover .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-popover .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-popover .bp3-popover-arrow-fill{
    fill:#ffffff; }
  .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3); }
  .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-popover-exit > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-popover .bp3-popover-content{
    position:relative;
    border-radius:3px; }
  .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
    max-width:350px;
    padding:20px; }
  .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
    width:350px; }
  .bp3-popover.bp3-minimal{
    margin:0 !important; }
    .bp3-popover.bp3-minimal .bp3-popover-arrow{
      display:none; }
    .bp3-popover.bp3-minimal.bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
      .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
        -webkit-transition-delay:0;
                transition-delay:0; }
      .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
        -webkit-transition-delay:0;
                transition-delay:0; }
  .bp3-popover.bp3-dark,
  .bp3-dark .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-popover .bp3-popover-content{
      background:#30404d;
      color:inherit; }
    .bp3-popover.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
      fill:#30404d; }

.bp3-popover-arrow::before{
  display:block;
  position:absolute;
  -webkit-transform:rotate(45deg);
          transform:rotate(45deg);
  border-radius:2px;
  content:""; }

.bp3-tether-pinned .bp3-popover-arrow{
  display:none; }

.bp3-popover-backdrop{
  background:rgba(255, 255, 255, 0); }

.bp3-transition-container{
  opacity:1;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  z-index:20; }
  .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
    opacity:0; }
  .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
    opacity:1;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-transition-container.bp3-popover-exit{
    opacity:1; }
  .bp3-transition-container.bp3-popover-exit-active{
    opacity:0;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-transition-container:focus{
    outline:none; }
  .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
    pointer-events:none; }
  .bp3-transition-container[data-x-out-of-boundaries]{
    display:none; }

span.bp3-popover-target{
  display:inline-block; }

.bp3-popover-wrapper.bp3-fill{
  width:100%; }

.bp3-portal{
  position:absolute;
  top:0;
  right:0;
  left:0; }
@-webkit-keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }
@keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }

.bp3-progress-bar{
  display:block;
  position:relative;
  border-radius:40px;
  background:rgba(92, 112, 128, 0.2);
  width:100%;
  height:8px;
  overflow:hidden; }
  .bp3-progress-bar .bp3-progress-meter{
    position:absolute;
    border-radius:40px;
    background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
    background-color:rgba(92, 112, 128, 0.8);
    background-size:30px 30px;
    width:100%;
    height:100%;
    -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
    animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
  .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
    background-image:none; }

.bp3-dark .bp3-progress-bar{
  background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-progress-bar .bp3-progress-meter{
    background-color:#8a9ba8; }

.bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
  background-color:#137cbd; }

.bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
  background-color:#0f9960; }

.bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
  background-color:#d9822b; }

.bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
  background-color:#db3737; }
@-webkit-keyframes skeleton-glow{
  from{
    border-color:rgba(206, 217, 224, 0.2);
    background:rgba(206, 217, 224, 0.2); }
  to{
    border-color:rgba(92, 112, 128, 0.2);
    background:rgba(92, 112, 128, 0.2); } }
@keyframes skeleton-glow{
  from{
    border-color:rgba(206, 217, 224, 0.2);
    background:rgba(206, 217, 224, 0.2); }
  to{
    border-color:rgba(92, 112, 128, 0.2);
    background:rgba(92, 112, 128, 0.2); } }
.bp3-skeleton{
  border-color:rgba(206, 217, 224, 0.2) !important;
  border-radius:2px;
  -webkit-box-shadow:none !important;
          box-shadow:none !important;
  background:rgba(206, 217, 224, 0.2);
  background-clip:padding-box !important;
  cursor:default;
  color:transparent !important;
  -webkit-animation:1000ms linear infinite alternate skeleton-glow;
          animation:1000ms linear infinite alternate skeleton-glow;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-skeleton::before, .bp3-skeleton::after,
  .bp3-skeleton *{
    visibility:hidden !important; }
.bp3-slider{
  width:100%;
  min-width:150px;
  height:40px;
  position:relative;
  outline:none;
  cursor:default;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-slider:hover{
    cursor:pointer; }
  .bp3-slider:active{
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-slider.bp3-disabled{
    opacity:0.5;
    cursor:not-allowed; }
  .bp3-slider.bp3-slider-unlabeled{
    height:16px; }

.bp3-slider-track,
.bp3-slider-progress{
  top:5px;
  right:0;
  left:0;
  height:6px;
  position:absolute; }

.bp3-slider-track{
  border-radius:3px;
  overflow:hidden; }

.bp3-slider-progress{
  background:rgba(92, 112, 128, 0.2); }
  .bp3-dark .bp3-slider-progress{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-slider-progress.bp3-intent-primary{
    background-color:#137cbd; }
  .bp3-slider-progress.bp3-intent-success{
    background-color:#0f9960; }
  .bp3-slider-progress.bp3-intent-warning{
    background-color:#d9822b; }
  .bp3-slider-progress.bp3-intent-danger{
    background-color:#db3737; }

.bp3-slider-handle{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  color:#182026;
  position:absolute;
  top:0;
  left:0;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
  cursor:pointer;
  width:16px;
  height:16px; }
  .bp3-slider-handle:hover{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5; }
  .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none; }
  .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
    outline:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
    .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }
  .bp3-slider-handle:focus{
    z-index:1; }
  .bp3-slider-handle:hover{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5;
    z-index:2;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:-webkit-grab;
    cursor:grab; }
  .bp3-slider-handle.bp3-active{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-disabled .bp3-slider-handle{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:#bfccd6;
    pointer-events:none; }
  .bp3-dark .bp3-slider-handle{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#30404d; }
    .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#202b33;
      background-image:none; }
    .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
      background-color:#394b59; }
    .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#293742; }
  .bp3-dark .bp3-disabled .bp3-slider-handle{
    border-color:#5c7080;
    -webkit-box-shadow:none;
            box-shadow:none;
    background:#5c7080; }
  .bp3-slider-handle .bp3-slider-label{
    margin-left:8px;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    background:#394b59;
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
      background:#e1e8ed;
      color:#394b59; }
    .bp3-disabled .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
    width:8px; }
  .bp3-slider-handle.bp3-start{
    border-top-right-radius:0;
    border-bottom-right-radius:0; }
  .bp3-slider-handle.bp3-end{
    margin-left:8px;
    border-top-left-radius:0;
    border-bottom-left-radius:0; }
    .bp3-slider-handle.bp3-end .bp3-slider-label{
      margin-left:0; }

.bp3-slider-label{
  -webkit-transform:translate(-50%, 20px);
          transform:translate(-50%, 20px);
  display:inline-block;
  position:absolute;
  padding:2px 5px;
  vertical-align:top;
  line-height:1;
  font-size:12px; }

.bp3-slider.bp3-vertical{
  width:40px;
  min-width:40px;
  height:150px; }
  .bp3-slider.bp3-vertical .bp3-slider-track,
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:0;
    bottom:0;
    left:5px;
    width:6px;
    height:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-label{
    -webkit-transform:translate(20px, 50%);
            transform:translate(20px, 50%); }
  .bp3-slider.bp3-vertical .bp3-slider-handle{
    top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
      margin-top:-8px;
      margin-left:0; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      margin-left:0;
      width:16px;
      height:8px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      border-top-left-radius:0;
      border-bottom-right-radius:3px; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
        -webkit-transform:translate(20px);
                transform:translate(20px); }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
      margin-bottom:8px;
      border-top-left-radius:3px;
      border-bottom-left-radius:0;
      border-bottom-right-radius:0; }

@-webkit-keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

@keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

.bp3-spinner{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  overflow:visible;
  vertical-align:middle; }
  .bp3-spinner svg{
    display:block; }
  .bp3-spinner path{
    fill-opacity:0; }
  .bp3-spinner .bp3-spinner-head{
    -webkit-transform-origin:center;
            transform-origin:center;
    -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    stroke:rgba(92, 112, 128, 0.8);
    stroke-linecap:round; }
  .bp3-spinner .bp3-spinner-track{
    stroke:rgba(92, 112, 128, 0.2); }

.bp3-spinner-animation{
  -webkit-animation:pt-spinner-animation 500ms linear infinite;
          animation:pt-spinner-animation 500ms linear infinite; }
  .bp3-no-spin > .bp3-spinner-animation{
    -webkit-animation:none;
            animation:none; }

.bp3-dark .bp3-spinner .bp3-spinner-head{
  stroke:#8a9ba8; }

.bp3-dark .bp3-spinner .bp3-spinner-track{
  stroke:rgba(16, 22, 26, 0.5); }

.bp3-spinner.bp3-intent-primary .bp3-spinner-head{
  stroke:#137cbd; }

.bp3-spinner.bp3-intent-success .bp3-spinner-head{
  stroke:#0f9960; }

.bp3-spinner.bp3-intent-warning .bp3-spinner-head{
  stroke:#d9822b; }

.bp3-spinner.bp3-intent-danger .bp3-spinner-head{
  stroke:#db3737; }
.bp3-tabs.bp3-vertical{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-tabs.bp3-vertical > .bp3-tab-list{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
      border-radius:3px;
      width:100%;
      padding:0 10px; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
        -webkit-box-shadow:none;
                box-shadow:none;
        background-color:rgba(19, 124, 189, 0.2); }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      top:0;
      right:0;
      bottom:0;
      left:0;
      border-radius:3px;
      background-color:rgba(19, 124, 189, 0.2);
      height:auto; }
  .bp3-tabs.bp3-vertical > .bp3-tab-panel{
    margin-top:0;
    padding-left:20px; }

.bp3-tab-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  -webkit-box-align:end;
      -ms-flex-align:end;
          align-items:flex-end;
  position:relative;
  margin:0;
  border:none;
  padding:0;
  list-style:none; }
  .bp3-tab-list > *:not(:last-child){
    margin-right:20px; }

.bp3-tab{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  position:relative;
  cursor:pointer;
  max-width:100%;
  vertical-align:top;
  line-height:30px;
  color:#182026;
  font-size:14px; }
  .bp3-tab a{
    display:block;
    text-decoration:none;
    color:inherit; }
  .bp3-tab-indicator-wrapper ~ .bp3-tab{
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    background-color:transparent !important; }
  .bp3-tab[aria-disabled="true"]{
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-tab[aria-selected="true"]{
    border-radius:0;
    -webkit-box-shadow:inset 0 -3px 0 #106ba3;
            box-shadow:inset 0 -3px 0 #106ba3; }
  .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
    color:#106ba3; }
  .bp3-tab:focus{
    -moz-outline-radius:0; }
  .bp3-large > .bp3-tab{
    line-height:40px;
    font-size:16px; }

.bp3-tab-panel{
  margin-top:20px; }
  .bp3-tab-panel[aria-hidden="true"]{
    display:none; }

.bp3-tab-indicator-wrapper{
  position:absolute;
  top:0;
  left:0;
  -webkit-transform:translateX(0), translateY(0);
          transform:translateX(0), translateY(0);
  -webkit-transition:height, width, -webkit-transform;
  transition:height, width, -webkit-transform;
  transition:height, transform, width;
  transition:height, transform, width, -webkit-transform;
  -webkit-transition-duration:200ms;
          transition-duration:200ms;
  -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
          transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
  pointer-events:none; }
  .bp3-tab-indicator-wrapper .bp3-tab-indicator{
    position:absolute;
    right:0;
    bottom:0;
    left:0;
    background-color:#106ba3;
    height:3px; }
  .bp3-tab-indicator-wrapper.bp3-no-animation{
    -webkit-transition:none;
    transition:none; }

.bp3-dark .bp3-tab{
  color:#f5f8fa; }
  .bp3-dark .bp3-tab[aria-disabled="true"]{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tab[aria-selected="true"]{
    -webkit-box-shadow:inset 0 -3px 0 #48aff0;
            box-shadow:inset 0 -3px 0 #48aff0; }
  .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
    color:#48aff0; }

.bp3-dark .bp3-tab-indicator{
  background-color:#48aff0; }

.bp3-flex-expander{
  -webkit-box-flex:1;
      -ms-flex:1 1;
          flex:1 1; }
.bp3-tag{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  position:relative;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:none;
          box-shadow:none;
  background-color:#5c7080;
  min-width:20px;
  max-width:100%;
  min-height:20px;
  padding:2px 6px;
  line-height:16px;
  color:#f5f8fa;
  font-size:12px; }
  .bp3-tag.bp3-interactive{
    cursor:pointer; }
    .bp3-tag.bp3-interactive:hover{
      background-color:rgba(92, 112, 128, 0.85); }
    .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
      background-color:rgba(92, 112, 128, 0.7); }
  .bp3-tag > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag::before,
  .bp3-tag > *{
    margin-right:4px; }
  .bp3-tag:empty::before,
  .bp3-tag > :last-child{
    margin-right:0; }
  .bp3-tag:focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:0;
    -moz-outline-radius:6px; }
  .bp3-tag.bp3-round{
    border-radius:30px;
    padding-right:8px;
    padding-left:8px; }
  .bp3-dark .bp3-tag{
    background-color:#bfccd6;
    color:#182026; }
    .bp3-dark .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-dark .bp3-tag.bp3-interactive:hover{
        background-color:rgba(191, 204, 214, 0.85); }
      .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
        background-color:rgba(191, 204, 214, 0.7); }
    .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
      fill:currentColor; }
  .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
    fill:#ffffff; }
  .bp3-tag.bp3-large,
  .bp3-large .bp3-tag{
    min-width:30px;
    min-height:30px;
    padding:0 10px;
    line-height:20px;
    font-size:14px; }
    .bp3-tag.bp3-large::before,
    .bp3-tag.bp3-large > *,
    .bp3-large .bp3-tag::before,
    .bp3-large .bp3-tag > *{
      margin-right:7px; }
    .bp3-tag.bp3-large:empty::before,
    .bp3-tag.bp3-large > :last-child,
    .bp3-large .bp3-tag:empty::before,
    .bp3-large .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag.bp3-large.bp3-round,
    .bp3-large .bp3-tag.bp3-round{
      padding-right:12px;
      padding-left:12px; }
  .bp3-tag.bp3-intent-primary{
    background:#137cbd;
    color:#ffffff; }
    .bp3-tag.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.85); }
      .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.7); }
  .bp3-tag.bp3-intent-success{
    background:#0f9960;
    color:#ffffff; }
    .bp3-tag.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.85); }
      .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.7); }
  .bp3-tag.bp3-intent-warning{
    background:#d9822b;
    color:#ffffff; }
    .bp3-tag.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.85); }
      .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.7); }
  .bp3-tag.bp3-intent-danger{
    background:#db3737;
    color:#ffffff; }
    .bp3-tag.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.85); }
      .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.7); }
  .bp3-tag.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
    fill:#5c7080; }
  .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
    background-color:rgba(138, 155, 168, 0.2);
    color:#182026; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.3); }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.4); }
    .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      color:#f5f8fa; }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.3); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
        fill:#a7b6c2; }
  .bp3-tag.bp3-minimal.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15);
    color:#106ba3; }
    .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
      fill:#137cbd; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25);
      color:#48aff0; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15);
    color:#0d8050; }
    .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
      fill:#0f9960; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25);
      color:#3dcc91; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15);
    color:#bf7326; }
    .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
      fill:#d9822b; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25);
      color:#ffb366; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15);
    color:#c23030; }
    .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
      fill:#db3737; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25);
      color:#ff7373; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.45); }

.bp3-tag-remove{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  opacity:0.5;
  margin-top:-2px;
  margin-right:-6px !important;
  margin-bottom:-2px;
  border:none;
  background:none;
  cursor:pointer;
  padding:2px;
  padding-left:0;
  color:inherit; }
  .bp3-tag-remove:hover{
    opacity:0.8;
    background:none;
    text-decoration:none; }
  .bp3-tag-remove:active{
    opacity:1; }
  .bp3-tag-remove:empty::before{
    line-height:1;
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-weight:400;
    font-style:normal;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-large .bp3-tag-remove{
    margin-right:-10px !important;
    padding:5px;
    padding-left:0; }
    .bp3-large .bp3-tag-remove:empty::before{
      line-height:1;
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-weight:400;
      font-style:normal; }
.bp3-tag-input{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  cursor:text;
  height:auto;
  min-height:30px;
  padding-right:0;
  padding-left:5px;
  line-height:inherit; }
  .bp3-tag-input > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag-input > .bp3-tag-input-values{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag-input .bp3-tag-input-icon{
    margin-top:7px;
    margin-right:7px;
    margin-left:2px;
    color:#5c7080; }
  .bp3-tag-input .bp3-tag-input-values{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -ms-flex-item-align:stretch;
        align-self:stretch;
    margin-top:5px;
    margin-right:7px;
    min-width:0; }
    .bp3-tag-input .bp3-tag-input-values > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-values::before,
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-right:5px; }
    .bp3-tag-input .bp3-tag-input-values:empty::before,
    .bp3-tag-input .bp3-tag-input-values > :last-child{
      margin-right:0; }
    .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
      padding-left:5px; }
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-bottom:5px; }
  .bp3-tag-input .bp3-tag{
    overflow-wrap:break-word; }
    .bp3-tag-input .bp3-tag.bp3-active{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
  .bp3-tag-input .bp3-input-ghost{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:80px;
    line-height:20px; }
    .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
      cursor:not-allowed; }
  .bp3-tag-input .bp3-button,
  .bp3-tag-input .bp3-spinner{
    margin:3px;
    margin-left:0; }
  .bp3-tag-input .bp3-button{
    min-width:24px;
    min-height:24px;
    padding:0 7px; }
  .bp3-tag-input.bp3-large{
    height:auto;
    min-height:40px; }
    .bp3-tag-input.bp3-large::before,
    .bp3-tag-input.bp3-large > *{
      margin-right:10px; }
    .bp3-tag-input.bp3-large:empty::before,
    .bp3-tag-input.bp3-large > :last-child{
      margin-right:0; }
    .bp3-tag-input.bp3-large .bp3-tag-input-icon{
      margin-top:10px;
      margin-left:5px; }
    .bp3-tag-input.bp3-large .bp3-input-ghost{
      line-height:30px; }
    .bp3-tag-input.bp3-large .bp3-button{
      min-width:30px;
      min-height:30px;
      padding:5px 10px;
      margin:5px;
      margin-left:0; }
    .bp3-tag-input.bp3-large .bp3-spinner{
      margin:8px;
      margin-left:0; }
  .bp3-tag-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
    background-color:#ffffff; }
    .bp3-tag-input.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
    color:#f5f8fa; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
      color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background-color:rgba(16, 22, 26, 0.3); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-input-ghost{
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  background:none;
  padding:0; }
  .bp3-input-ghost::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost:focus{
    outline:none !important; }
.bp3-toast{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  position:relative !important;
  margin:20px 0 0;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  background-color:#ffffff;
  min-width:300px;
  max-width:500px;
  pointer-events:all; }
  .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-toast.bp3-toast-exit{
    opacity:1;
    -webkit-filter:blur(0);
            filter:blur(0); }
  .bp3-toast.bp3-toast-exit-active{
    opacity:0;
    -webkit-filter:blur(10px);
            filter:blur(10px);
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:opacity, filter;
    transition-property:opacity, filter, -webkit-filter;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-toast.bp3-toast-exit ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0); }
  .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:50ms;
            transition-delay:50ms; }
  .bp3-toast .bp3-button-group{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:5px;
    padding-left:0; }
  .bp3-toast > .bp3-icon{
    margin:12px;
    margin-right:0;
    color:#5c7080; }
  .bp3-toast.bp3-dark,
  .bp3-dark .bp3-toast{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
    background-color:#394b59; }
    .bp3-toast.bp3-dark > .bp3-icon,
    .bp3-dark .bp3-toast > .bp3-icon{
      color:#a7b6c2; }
  .bp3-toast[class*="bp3-intent-"] a{
    color:rgba(255, 255, 255, 0.7); }
    .bp3-toast[class*="bp3-intent-"] a:hover{
      color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
    color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
  .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    color:rgba(255, 255, 255, 0.7) !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
    outline-color:rgba(255, 255, 255, 0.5); }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
    background-color:rgba(255, 255, 255, 0.15) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    background-color:rgba(255, 255, 255, 0.3) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
    background:rgba(255, 255, 255, 0.3) !important; }
  .bp3-toast.bp3-intent-primary{
    background-color:#137cbd;
    color:#ffffff; }
  .bp3-toast.bp3-intent-success{
    background-color:#0f9960;
    color:#ffffff; }
  .bp3-toast.bp3-intent-warning{
    background-color:#d9822b;
    color:#ffffff; }
  .bp3-toast.bp3-intent-danger{
    background-color:#db3737;
    color:#ffffff; }

.bp3-toast-message{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  padding:11px;
  word-break:break-word; }

.bp3-toast-container{
  display:-webkit-box !important;
  display:-ms-flexbox !important;
  display:flex !important;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  position:fixed;
  right:0;
  left:0;
  z-index:40;
  overflow:hidden;
  padding:0 20px 20px;
  pointer-events:none; }
  .bp3-toast-container.bp3-toast-container-top{
    top:0;
    bottom:auto; }
  .bp3-toast-container.bp3-toast-container-bottom{
    -webkit-box-orient:vertical;
    -webkit-box-direction:reverse;
        -ms-flex-direction:column-reverse;
            flex-direction:column-reverse;
    top:auto;
    bottom:0; }
  .bp3-toast-container.bp3-toast-container-left{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
  .bp3-toast-container.bp3-toast-container-right{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end; }

.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
  -webkit-transform:translateY(60px);
          transform:translateY(60px); }
.bp3-tooltip{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1); }
  .bp3-tooltip .bp3-popover-arrow{
    position:absolute;
    width:22px;
    height:22px; }
    .bp3-tooltip .bp3-popover-arrow::before{
      margin:4px;
      width:14px;
      height:14px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
    margin-top:-11px;
    margin-bottom:11px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-8px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
    margin-left:11px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      left:-8px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
    margin-top:11px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      top:-8px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
    margin-right:11px;
    margin-left:-11px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      right:-8px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
    top:-0.22183px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
    right:-0.22183px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
    left:-0.22183px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
    bottom:-0.22183px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-tooltip .bp3-popover-content{
    background:#394b59;
    color:#f5f8fa; }
  .bp3-tooltip .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-tooltip .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-tooltip .bp3-popover-arrow-fill{
    fill:#394b59; }
  .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8); }
  .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-popover-exit > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-tooltip .bp3-popover-content{
    padding:10px 12px; }
  .bp3-tooltip.bp3-dark,
  .bp3-dark .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-tooltip .bp3-popover-content{
      background:#e1e8ed;
      color:#394b59; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#e1e8ed; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
    background:#137cbd;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
    fill:#137cbd; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-content{
    background:#0f9960;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
    fill:#0f9960; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
    background:#d9822b;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
    fill:#d9822b; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
    background:#db3737;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
    fill:#db3737; }

.bp3-tooltip-indicator{
  border-bottom:dotted 1px;
  cursor:help; }
.bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
  color:#5c7080; }
  .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-tree-node-list{
  margin:0;
  padding-left:0;
  list-style:none; }

.bp3-tree-root{
  position:relative;
  background-color:transparent;
  cursor:default;
  padding-left:0; }

.bp3-tree-node-content-0{
  padding-left:0px; }

.bp3-tree-node-content-1{
  padding-left:23px; }

.bp3-tree-node-content-2{
  padding-left:46px; }

.bp3-tree-node-content-3{
  padding-left:69px; }

.bp3-tree-node-content-4{
  padding-left:92px; }

.bp3-tree-node-content-5{
  padding-left:115px; }

.bp3-tree-node-content-6{
  padding-left:138px; }

.bp3-tree-node-content-7{
  padding-left:161px; }

.bp3-tree-node-content-8{
  padding-left:184px; }

.bp3-tree-node-content-9{
  padding-left:207px; }

.bp3-tree-node-content-10{
  padding-left:230px; }

.bp3-tree-node-content-11{
  padding-left:253px; }

.bp3-tree-node-content-12{
  padding-left:276px; }

.bp3-tree-node-content-13{
  padding-left:299px; }

.bp3-tree-node-content-14{
  padding-left:322px; }

.bp3-tree-node-content-15{
  padding-left:345px; }

.bp3-tree-node-content-16{
  padding-left:368px; }

.bp3-tree-node-content-17{
  padding-left:391px; }

.bp3-tree-node-content-18{
  padding-left:414px; }

.bp3-tree-node-content-19{
  padding-left:437px; }

.bp3-tree-node-content-20{
  padding-left:460px; }

.bp3-tree-node-content{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  width:100%;
  height:30px;
  padding-right:5px; }
  .bp3-tree-node-content:hover{
    background-color:rgba(191, 204, 214, 0.4); }

.bp3-tree-node-caret,
.bp3-tree-node-caret-none{
  min-width:30px; }

.bp3-tree-node-caret{
  color:#5c7080;
  -webkit-transform:rotate(0deg);
          transform:rotate(0deg);
  cursor:pointer;
  padding:7px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tree-node-caret:hover{
    color:#182026; }
  .bp3-dark .bp3-tree-node-caret{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree-node-caret:hover{
      color:#f5f8fa; }
  .bp3-tree-node-caret.bp3-tree-node-caret-open{
    -webkit-transform:rotate(90deg);
            transform:rotate(90deg); }
  .bp3-tree-node-caret.bp3-icon-standard::before{
    content:""; }

.bp3-tree-node-icon{
  position:relative;
  margin-right:7px; }

.bp3-tree-node-label{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-label span{
    display:inline; }

.bp3-tree-node-secondary-label{
  padding:0 5px;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-secondary-label .bp3-popover-wrapper,
  .bp3-tree-node-secondary-label .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-content{
  background-color:inherit;
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
.bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
    color:#ffffff; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
    color:rgba(255, 255, 255, 0.7); }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
    color:#ffffff; }

.bp3-dark .bp3-tree-node-content:hover{
  background-color:rgba(92, 112, 128, 0.3); }

.bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
  color:#a7b6c2; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
/*!

Copyright 2017-present Palantir Technologies, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.

*/
.bp3-omnibar{
  -webkit-filter:blur(0);
          filter:blur(0);
  opacity:1;
  top:20vh;
  left:calc(50% - 250px);
  z-index:21;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  background-color:#ffffff;
  width:500px; }
  .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2; }
  .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-omnibar.bp3-overlay-exit{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1; }
  .bp3-omnibar.bp3-overlay-exit-active{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-omnibar .bp3-input{
    border-radius:0;
    background-color:transparent; }
    .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-omnibar .bp3-menu{
    border-radius:0;
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    background-color:transparent;
    max-height:calc(60vh - 40px);
    overflow:auto; }
    .bp3-omnibar .bp3-menu:empty{
      display:none; }
  .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    background-color:#30404d; }

.bp3-omnibar-overlay .bp3-overlay-backdrop{
  background-color:rgba(16, 22, 26, 0.2); }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-width:400px;
  max-height:300px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }

.bp3-multi-select{
  min-width:150px; }

.bp3-multi-select-popover .bp3-menu{
  max-width:400px;
  max-height:300px;
  overflow:auto; }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-width:400px;
  max-height:300px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDhoLTIuODFjLS40NS0uNzgtMS4wNy0xLjQ1LTEuODItMS45NkwxNyA0LjQxIDE1LjU5IDNsLTIuMTcgMi4xN0MxMi45NiA1LjA2IDEyLjQ5IDUgMTIgNWMtLjQ5IDAtLjk2LjA2LTEuNDEuMTdMOC40MSAzIDcgNC40MWwxLjYyIDEuNjNDNy44OCA2LjU1IDcuMjYgNy4yMiA2LjgxIDhINHYyaDIuMDljLS4wNS4zMy0uMDkuNjYtLjA5IDF2MUg0djJoMnYxYzAgLjM0LjA0LjY3LjA5IDFINHYyaDIuODFjMS4wNCAxLjc5IDIuOTcgMyA1LjE5IDNzNC4xNS0xLjIxIDUuMTktM0gyMHYtMmgtMi4wOWMuMDUtLjMzLjA5LS42Ni4wOS0xdi0xaDJ2LTJoLTJ2LTFjMC0uMzQtLjA0LS42Ny0uMDktMUgyMFY4em0tNiA4aC00di0yaDR2MnptMC00aC00di0yaDR2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTYuMTdMNC44MyAxMmwtMS40MiAxLjQxTDkgMTkgMjEgN2wtMS40MS0xLjQxeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJDYXBhXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB2aWV3Qm94PSIwIDAgNTAuOTc4IDUwLjk3OCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgNTAuOTc4IDUwLjk3ODsiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxnPg0KCQkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik00My41Miw3LjQ1OEMzOC43MTEsMi42NDgsMzIuMzA3LDAsMjUuNDg5LDBDMTguNjcsMCwxMi4yNjYsMi42NDgsNy40NTgsNy40NTgNCgkJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDANCgkJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoNCgkJCQkgTTQyLjEwNiw0Mi4xMDVjLTQuNDMyLDQuNDMxLTEwLjMzMiw2Ljg3Mi0xNi42MTUsNi44NzJoLTAuMDAyYy02LjI4NS0wLjAwMS0xMi4xODctMi40NDEtMTYuNjE3LTYuODcyDQoJCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzINCgkJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4NCgkJPC9nPg0KCQk8Zz4NCgkJCTxwYXRoIHN0eWxlPSJmaWxsOiMwMTAwMDI7IiBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1Mw0KCQkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUNCgkJCQljMC0xLjA5Ni0wLjI2LTIuMDg4LTAuNzc5LTIuOTc5Yy0wLjU2NS0wLjg3OS0xLjUwMS0xLjMzNi0yLjgwNi0xLjM2OWMtMS44MDIsMC4wNTctMi45ODUsMC42NjctMy41NSwxLjgzMg0KCQkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkNCgkJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQ0KCQkJCWMwLDEuMTQyLTAuMTM3LDIuMTExLTAuNDEsMi45MTFjLTAuMzA5LDAuODQ1LTAuNzMxLDEuNTkzLTEuMjY4LDIuMjQzYy0wLjQ5MiwwLjY1LTEuMDY4LDEuMzE4LTEuNzMsMi4wMDINCgkJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5DQoJCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+DQoJCTwvZz4NCgk8L2c+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}
.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}
.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}
.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}
.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}
.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}
.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}
.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}
.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}
.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}
.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}
.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}
.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}
.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}
.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}
.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}
.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}
.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}
.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}
.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}
.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}
.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}
.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}
.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}
.jp-FileIcon {
  background-image: var(--jp-icon-file);
}
.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}
.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}
.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}
.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}
.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}
.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}
.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}
.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}
.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}
.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}
.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}
.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}
.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}
.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}
.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}
.jp-ListIcon {
  background-image: var(--jp-icon-list);
}
.jp-ListingsInfoIcon {
  background-image: var(--jp-icon-listings-info);
}
.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}
.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}
.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}
.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}
.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}
.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}
.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}
.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}
.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}
.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}
.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}
.jp-RunIcon {
  background-image: var(--jp-icon-run);
}
.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}
.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}
.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}
.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}
.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}
.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}
.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}
.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}
.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}
.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}
.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}
.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}
.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

:root {
  --jp-icon-search-white: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
}

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}
/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}
/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}
.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}
.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}
.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}
.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}
.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}
.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}
.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}
.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}
/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}
.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}
.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}
.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}
.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}
.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}
.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}
/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

/* CSS for icons in selected items in the settings editor */
#setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
#setting-editor
  .jp-PluginList
  .jp-mod-selected
  .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected tabs in the sidebar tab manager */
#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
  fill: #fff;
}

#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable[fill] {
  fill: var(--jp-brand-color1);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable-inverse[fill] {
  fill: #fff;
}

/**
 * TODO: come up with non css-hack solution for showing the busy icon on top
 *  of the close icon
 * CSS for complex behavior of close icon of tabs in the sidebar tab manager
 */
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: #fff;
}

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) svg {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

/* Override Blueprint's _reset.scss styles */
html {
  box-sizing: unset;
}

*,
*::before,
*::after {
  box-sizing: unset;
}

body {
  color: unset;
  font-family: var(--jp-ui-font-family);
}

p {
  margin-top: unset;
  margin-bottom: unset;
}

small {
  font-size: unset;
}

strong {
  font-weight: unset;
}

/* Override Blueprint's _typography.scss styles */
a {
  text-decoration: unset;
  color: unset;
}
a:hover {
  text-decoration: unset;
  color: unset;
}

/* Override Blueprint's _accessibility.scss styles */
:focus {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

/* Styles for ui-components */
.jp-Button {
  border-radius: var(--jp-border-radius);
  padding: 0px 12px;
  font-size: var(--jp-ui-font-size1);
}

/* Use our own theme for hover styles */
button.jp-Button.bp3-button.bp3-minimal:hover {
  background-color: var(--jp-layout-color2);
}
.jp-Button.minimal {
  color: unset !important;
}

.jp-Button.jp-ToolbarButtonComponent {
  text-transform: none;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color3);
}

.jp-BPIcon {
  display: inline-block;
  vertical-align: middle;
  margin: auto;
}

/* Stop blueprint futzing with our icon fills */
.bp3-icon.jp-BPIcon > svg:not([fill]) {
  fill: var(--jp-inverse-layout-color3);
}

.jp-InputGroupAction {
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

/* Use our own theme for hover and option styles */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}
select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-Collapse-header {
  padding: 1px 12px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size2);
}

.jp-Collapse-header:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Collapse-contents {
  padding: 0px 12px 0px 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0px;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0px 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.lm-CommandPalette-wrapper::after {
  content: ' ';
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  height: 30px;
  width: 10px;
  padding: 0px 10px;
  background-image: var(--jp-icon-search-white);
  background-size: 20px;
  background-repeat: no-repeat;
  background-position: center;
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color3);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0px;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color3);
}

.lm-CommandPalette-item.lm-mod-active {
  background: var(--jp-layout-color3);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  background: var(--jp-layout-color4);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color3);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.4;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty:after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0px 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0px;
  left: 0px;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px;
  padding-bottom: 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0px;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

.jp-Dialog-header {
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0px 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

.jp-HoverBox.jp-mod-outofview {
  display: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent:before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent:after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0px 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  height: 28px;
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  background-color: var(--jp-layout-color1);
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  height: 32px;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 1;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px;
  margin: 0px;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px 6px;
  margin: 0px;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent span {
  padding: 0px;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
body.lm-mod-override-cursor * {
  cursor: inherit !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* BASICS */

.CodeMirror {
  /* Set height, width, borders, and global font properties here */
  font-family: monospace;
  height: 300px;
  color: black;
  direction: ltr;
}

/* PADDING */

.CodeMirror-lines {
  padding: 4px 0; /* Vertical padding around content */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  padding: 0 4px; /* Horizontal padding of content */
}

.CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  background-color: white; /* The little square between H and V scrollbars */
}

/* GUTTER */

.CodeMirror-gutters {
  border-right: 1px solid #ddd;
  background-color: #f7f7f7;
  white-space: nowrap;
}
.CodeMirror-linenumbers {}
.CodeMirror-linenumber {
  padding: 0 3px 0 5px;
  min-width: 20px;
  text-align: right;
  color: #999;
  white-space: nowrap;
}

.CodeMirror-guttermarker { color: black; }
.CodeMirror-guttermarker-subtle { color: #999; }

/* CURSOR */

.CodeMirror-cursor {
  border-left: 1px solid black;
  border-right: none;
  width: 0;
}
/* Shown when moving in bi-directional text */
.CodeMirror div.CodeMirror-secondarycursor {
  border-left: 1px solid silver;
}
.cm-fat-cursor .CodeMirror-cursor {
  width: auto;
  border: 0 !important;
  background: #7e7;
}
.cm-fat-cursor div.CodeMirror-cursors {
  z-index: 1;
}
.cm-fat-cursor-mark {
  background-color: rgba(20, 255, 20, 0.5);
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
}
.cm-animate-fat-cursor {
  width: auto;
  border: 0;
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
  background-color: #7e7;
}
@-moz-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@-webkit-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}

/* Can style cursor different in overwrite (non-insert) mode */
.CodeMirror-overwrite .CodeMirror-cursor {}

.cm-tab { display: inline-block; text-decoration: inherit; }

.CodeMirror-rulers {
  position: absolute;
  left: 0; right: 0; top: -50px; bottom: 0;
  overflow: hidden;
}
.CodeMirror-ruler {
  border-left: 1px solid #ccc;
  top: 0; bottom: 0;
  position: absolute;
}

/* DEFAULT THEME */

.cm-s-default .cm-header {color: blue;}
.cm-s-default .cm-quote {color: #090;}
.cm-negative {color: #d44;}
.cm-positive {color: #292;}
.cm-header, .cm-strong {font-weight: bold;}
.cm-em {font-style: italic;}
.cm-link {text-decoration: underline;}
.cm-strikethrough {text-decoration: line-through;}

.cm-s-default .cm-keyword {color: #708;}
.cm-s-default .cm-atom {color: #219;}
.cm-s-default .cm-number {color: #164;}
.cm-s-default .cm-def {color: #00f;}
.cm-s-default .cm-variable,
.cm-s-default .cm-punctuation,
.cm-s-default .cm-property,
.cm-s-default .cm-operator {}
.cm-s-default .cm-variable-2 {color: #05a;}
.cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
.cm-s-default .cm-comment {color: #a50;}
.cm-s-default .cm-string {color: #a11;}
.cm-s-default .cm-string-2 {color: #f50;}
.cm-s-default .cm-meta {color: #555;}
.cm-s-default .cm-qualifier {color: #555;}
.cm-s-default .cm-builtin {color: #30a;}
.cm-s-default .cm-bracket {color: #997;}
.cm-s-default .cm-tag {color: #170;}
.cm-s-default .cm-attribute {color: #00c;}
.cm-s-default .cm-hr {color: #999;}
.cm-s-default .cm-link {color: #00c;}

.cm-s-default .cm-error {color: #f00;}
.cm-invalidchar {color: #f00;}

.CodeMirror-composing { border-bottom: 2px solid; }

/* Default styles for common addons */

div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
.CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
.CodeMirror-activeline-background {background: #e8f2ff;}

/* STOP */

/* The rest of this file contains styles related to the mechanics of
   the editor. You probably shouldn't touch them. */

.CodeMirror {
  position: relative;
  overflow: hidden;
  background: white;
}

.CodeMirror-scroll {
  overflow: scroll !important; /* Things will break if this is overridden */
  /* 30px is the magic margin used to hide the element's real scrollbars */
  /* See overflow: hidden in .CodeMirror */
  margin-bottom: -30px; margin-right: -30px;
  padding-bottom: 30px;
  height: 100%;
  outline: none; /* Prevent dragging from highlighting the element */
  position: relative;
}
.CodeMirror-sizer {
  position: relative;
  border-right: 30px solid transparent;
}

/* The fake, visible scrollbars. Used to force redraw during scrolling
   before actual scrolling happens, thus preventing shaking and
   flickering artifacts. */
.CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  position: absolute;
  z-index: 6;
  display: none;
}
.CodeMirror-vscrollbar {
  right: 0; top: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}
.CodeMirror-hscrollbar {
  bottom: 0; left: 0;
  overflow-y: hidden;
  overflow-x: scroll;
}
.CodeMirror-scrollbar-filler {
  right: 0; bottom: 0;
}
.CodeMirror-gutter-filler {
  left: 0; bottom: 0;
}

.CodeMirror-gutters {
  position: absolute; left: 0; top: 0;
  min-height: 100%;
  z-index: 3;
}
.CodeMirror-gutter {
  white-space: normal;
  height: 100%;
  display: inline-block;
  vertical-align: top;
  margin-bottom: -30px;
}
.CodeMirror-gutter-wrapper {
  position: absolute;
  z-index: 4;
  background: none !important;
  border: none !important;
}
.CodeMirror-gutter-background {
  position: absolute;
  top: 0; bottom: 0;
  z-index: 4;
}
.CodeMirror-gutter-elt {
  position: absolute;
  cursor: default;
  z-index: 4;
}
.CodeMirror-gutter-wrapper ::selection { background-color: transparent }
.CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }

.CodeMirror-lines {
  cursor: text;
  min-height: 1px; /* prevents collapsing before first draw */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  /* Reset some styles that the rest of the page might have set */
  -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
  border-width: 0;
  background: transparent;
  font-family: inherit;
  font-size: inherit;
  margin: 0;
  white-space: pre;
  word-wrap: normal;
  line-height: inherit;
  color: inherit;
  z-index: 2;
  position: relative;
  overflow: visible;
  -webkit-tap-highlight-color: transparent;
  -webkit-font-variant-ligatures: contextual;
  font-variant-ligatures: contextual;
}
.CodeMirror-wrap pre.CodeMirror-line,
.CodeMirror-wrap pre.CodeMirror-line-like {
  word-wrap: break-word;
  white-space: pre-wrap;
  word-break: normal;
}

.CodeMirror-linebackground {
  position: absolute;
  left: 0; right: 0; top: 0; bottom: 0;
  z-index: 0;
}

.CodeMirror-linewidget {
  position: relative;
  z-index: 2;
  padding: 0.1px; /* Force widget margins to stay inside of the container */
}

.CodeMirror-widget {}

.CodeMirror-rtl pre { direction: rtl; }

.CodeMirror-code {
  outline: none;
}

/* Force content-box sizing for the elements where we expect it */
.CodeMirror-scroll,
.CodeMirror-sizer,
.CodeMirror-gutter,
.CodeMirror-gutters,
.CodeMirror-linenumber {
  -moz-box-sizing: content-box;
  box-sizing: content-box;
}

.CodeMirror-measure {
  position: absolute;
  width: 100%;
  height: 0;
  overflow: hidden;
  visibility: hidden;
}

.CodeMirror-cursor {
  position: absolute;
  pointer-events: none;
}
.CodeMirror-measure pre { position: static; }

div.CodeMirror-cursors {
  visibility: hidden;
  position: relative;
  z-index: 3;
}
div.CodeMirror-dragcursors {
  visibility: visible;
}

.CodeMirror-focused div.CodeMirror-cursors {
  visibility: visible;
}

.CodeMirror-selected { background: #d9d9d9; }
.CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
.CodeMirror-crosshair { cursor: crosshair; }
.CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
.CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }

.cm-searching {
  background-color: #ffa;
  background-color: rgba(255, 255, 0, .4);
}

/* Used to force a border model for a node */
.cm-force-border { padding-right: .1px; }

@media print {
  /* Hide the cursor when printing */
  .CodeMirror div.CodeMirror-cursors {
    visibility: hidden;
  }
}

/* See issue #2901 */
.cm-tab-wrap-hack:after { content: ''; }

/* Help users use markselection to safely style text background */
span.CodeMirror-selectedtext { background: none; }

.CodeMirror-dialog {
  position: absolute;
  left: 0; right: 0;
  background: inherit;
  z-index: 15;
  padding: .1em .8em;
  overflow: hidden;
  color: inherit;
}

.CodeMirror-dialog-top {
  border-bottom: 1px solid #eee;
  top: 0;
}

.CodeMirror-dialog-bottom {
  border-top: 1px solid #eee;
  bottom: 0;
}

.CodeMirror-dialog input {
  border: none;
  outline: none;
  background: transparent;
  width: 20em;
  color: inherit;
  font-family: monospace;
}

.CodeMirror-dialog button {
  font-size: 70%;
}

.CodeMirror-foldmarker {
  color: blue;
  text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
  font-family: arial;
  line-height: .3;
  cursor: pointer;
}
.CodeMirror-foldgutter {
  width: .7em;
}
.CodeMirror-foldgutter-open,
.CodeMirror-foldgutter-folded {
  cursor: pointer;
}
.CodeMirror-foldgutter-open:after {
  content: "\25BE";
}
.CodeMirror-foldgutter-folded:after {
  content: "\25B8";
}

/*
  Name:       material
  Author:     Mattia Astorino (http://github.com/equinusocio)
  Website:    https://material-theme.site/
*/

.cm-s-material.CodeMirror {
  background-color: #263238;
  color: #EEFFFF;
}

.cm-s-material .CodeMirror-gutters {
  background: #263238;
  color: #546E7A;
  border: none;
}

.cm-s-material .CodeMirror-guttermarker,
.cm-s-material .CodeMirror-guttermarker-subtle,
.cm-s-material .CodeMirror-linenumber {
  color: #546E7A;
}

.cm-s-material .CodeMirror-cursor {
  border-left: 1px solid #FFCC00;
}

.cm-s-material div.CodeMirror-selected {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material.CodeMirror-focused div.CodeMirror-selected {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material .CodeMirror-line::selection,
.cm-s-material .CodeMirror-line>span::selection,
.cm-s-material .CodeMirror-line>span>span::selection {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material .CodeMirror-line::-moz-selection,
.cm-s-material .CodeMirror-line>span::-moz-selection,
.cm-s-material .CodeMirror-line>span>span::-moz-selection {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material .CodeMirror-activeline-background {
  background: rgba(0, 0, 0, 0.5);
}

.cm-s-material .cm-keyword {
  color: #C792EA;
}

.cm-s-material .cm-operator {
  color: #89DDFF;
}

.cm-s-material .cm-variable-2 {
  color: #EEFFFF;
}

.cm-s-material .cm-variable-3,
.cm-s-material .cm-type {
  color: #f07178;
}

.cm-s-material .cm-builtin {
  color: #FFCB6B;
}

.cm-s-material .cm-atom {
  color: #F78C6C;
}

.cm-s-material .cm-number {
  color: #FF5370;
}

.cm-s-material .cm-def {
  color: #82AAFF;
}

.cm-s-material .cm-string {
  color: #C3E88D;
}

.cm-s-material .cm-string-2 {
  color: #f07178;
}

.cm-s-material .cm-comment {
  color: #546E7A;
}

.cm-s-material .cm-variable {
  color: #f07178;
}

.cm-s-material .cm-tag {
  color: #FF5370;
}

.cm-s-material .cm-meta {
  color: #FFCB6B;
}

.cm-s-material .cm-attribute {
  color: #C792EA;
}

.cm-s-material .cm-property {
  color: #C792EA;
}

.cm-s-material .cm-qualifier {
  color: #DECB6B;
}

.cm-s-material .cm-variable-3,
.cm-s-material .cm-type {
  color: #DECB6B;
}


.cm-s-material .cm-error {
  color: rgba(255, 255, 255, 1.0);
  background-color: #FF5370;
}

.cm-s-material .CodeMirror-matchingbracket {
  text-decoration: underline;
  color: white !important;
}
/**
 * "
 *  Using Zenburn color palette from the Emacs Zenburn Theme
 *  https://github.com/bbatsov/zenburn-emacs/blob/master/zenburn-theme.el
 *
 *  Also using parts of https://github.com/xavi/coderay-lighttable-theme
 * "
 * From: https://github.com/wisenomad/zenburn-lighttable-theme/blob/master/zenburn.css
 */

.cm-s-zenburn .CodeMirror-gutters { background: #3f3f3f !important; }
.cm-s-zenburn .CodeMirror-foldgutter-open, .CodeMirror-foldgutter-folded { color: #999; }
.cm-s-zenburn .CodeMirror-cursor { border-left: 1px solid white; }
.cm-s-zenburn { background-color: #3f3f3f; color: #dcdccc; }
.cm-s-zenburn span.cm-builtin { color: #dcdccc; font-weight: bold; }
.cm-s-zenburn span.cm-comment { color: #7f9f7f; }
.cm-s-zenburn span.cm-keyword { color: #f0dfaf; font-weight: bold; }
.cm-s-zenburn span.cm-atom { color: #bfebbf; }
.cm-s-zenburn span.cm-def { color: #dcdccc; }
.cm-s-zenburn span.cm-variable { color: #dfaf8f; }
.cm-s-zenburn span.cm-variable-2 { color: #dcdccc; }
.cm-s-zenburn span.cm-string { color: #cc9393; }
.cm-s-zenburn span.cm-string-2 { color: #cc9393; }
.cm-s-zenburn span.cm-number { color: #dcdccc; }
.cm-s-zenburn span.cm-tag { color: #93e0e3; }
.cm-s-zenburn span.cm-property { color: #dfaf8f; }
.cm-s-zenburn span.cm-attribute { color: #dfaf8f; }
.cm-s-zenburn span.cm-qualifier { color: #7cb8bb; }
.cm-s-zenburn span.cm-meta { color: #f0dfaf; }
.cm-s-zenburn span.cm-header { color: #f0efd0; }
.cm-s-zenburn span.cm-operator { color: #f0efd0; }
.cm-s-zenburn span.CodeMirror-matchingbracket { box-sizing: border-box; background: transparent; border-bottom: 1px solid; }
.cm-s-zenburn span.CodeMirror-nonmatchingbracket { border-bottom: 1px solid; background: none; }
.cm-s-zenburn .CodeMirror-activeline { background: #000000; }
.cm-s-zenburn .CodeMirror-activeline-background { background: #000000; }
.cm-s-zenburn div.CodeMirror-selected { background: #545454; }
.cm-s-zenburn .CodeMirror-focused div.CodeMirror-selected { background: #4f4f4f; }

.cm-s-abcdef.CodeMirror { background: #0f0f0f; color: #defdef; }
.cm-s-abcdef div.CodeMirror-selected { background: #515151; }
.cm-s-abcdef .CodeMirror-line::selection, .cm-s-abcdef .CodeMirror-line > span::selection, .cm-s-abcdef .CodeMirror-line > span > span::selection { background: rgba(56, 56, 56, 0.99); }
.cm-s-abcdef .CodeMirror-line::-moz-selection, .cm-s-abcdef .CodeMirror-line > span::-moz-selection, .cm-s-abcdef .CodeMirror-line > span > span::-moz-selection { background: rgba(56, 56, 56, 0.99); }
.cm-s-abcdef .CodeMirror-gutters { background: #555; border-right: 2px solid #314151; }
.cm-s-abcdef .CodeMirror-guttermarker { color: #222; }
.cm-s-abcdef .CodeMirror-guttermarker-subtle { color: azure; }
.cm-s-abcdef .CodeMirror-linenumber { color: #FFFFFF; }
.cm-s-abcdef .CodeMirror-cursor { border-left: 1px solid #00FF00; }

.cm-s-abcdef span.cm-keyword { color: darkgoldenrod; font-weight: bold; }
.cm-s-abcdef span.cm-atom { color: #77F; }
.cm-s-abcdef span.cm-number { color: violet; }
.cm-s-abcdef span.cm-def { color: #fffabc; }
.cm-s-abcdef span.cm-variable { color: #abcdef; }
.cm-s-abcdef span.cm-variable-2 { color: #cacbcc; }
.cm-s-abcdef span.cm-variable-3, .cm-s-abcdef span.cm-type { color: #def; }
.cm-s-abcdef span.cm-property { color: #fedcba; }
.cm-s-abcdef span.cm-operator { color: #ff0; }
.cm-s-abcdef span.cm-comment { color: #7a7b7c; font-style: italic;}
.cm-s-abcdef span.cm-string { color: #2b4; }
.cm-s-abcdef span.cm-meta { color: #C9F; }
.cm-s-abcdef span.cm-qualifier { color: #FFF700; }
.cm-s-abcdef span.cm-builtin { color: #30aabc; }
.cm-s-abcdef span.cm-bracket { color: #8a8a8a; }
.cm-s-abcdef span.cm-tag { color: #FFDD44; }
.cm-s-abcdef span.cm-attribute { color: #DDFF00; }
.cm-s-abcdef span.cm-error { color: #FF0000; }
.cm-s-abcdef span.cm-header { color: aquamarine; font-weight: bold; }
.cm-s-abcdef span.cm-link { color: blueviolet; }

.cm-s-abcdef .CodeMirror-activeline-background { background: #314151; }

/*

    Name:       Base16 Default Light
    Author:     Chris Kempson (http://chriskempson.com)

    CodeMirror template by Jan T. Sott (https://github.com/idleberg/base16-codemirror)
    Original Base16 color scheme by Chris Kempson (https://github.com/chriskempson/base16)

*/

.cm-s-base16-light.CodeMirror { background: #f5f5f5; color: #202020; }
.cm-s-base16-light div.CodeMirror-selected { background: #e0e0e0; }
.cm-s-base16-light .CodeMirror-line::selection, .cm-s-base16-light .CodeMirror-line > span::selection, .cm-s-base16-light .CodeMirror-line > span > span::selection { background: #e0e0e0; }
.cm-s-base16-light .CodeMirror-line::-moz-selection, .cm-s-base16-light .CodeMirror-line > span::-moz-selection, .cm-s-base16-light .CodeMirror-line > span > span::-moz-selection { background: #e0e0e0; }
.cm-s-base16-light .CodeMirror-gutters { background: #f5f5f5; border-right: 0px; }
.cm-s-base16-light .CodeMirror-guttermarker { color: #ac4142; }
.cm-s-base16-light .CodeMirror-guttermarker-subtle { color: #b0b0b0; }
.cm-s-base16-light .CodeMirror-linenumber { color: #b0b0b0; }
.cm-s-base16-light .CodeMirror-cursor { border-left: 1px solid #505050; }

.cm-s-base16-light span.cm-comment { color: #8f5536; }
.cm-s-base16-light span.cm-atom { color: #aa759f; }
.cm-s-base16-light span.cm-number { color: #aa759f; }

.cm-s-base16-light span.cm-property, .cm-s-base16-light span.cm-attribute { color: #90a959; }
.cm-s-base16-light span.cm-keyword { color: #ac4142; }
.cm-s-base16-light span.cm-string { color: #f4bf75; }

.cm-s-base16-light span.cm-variable { color: #90a959; }
.cm-s-base16-light span.cm-variable-2 { color: #6a9fb5; }
.cm-s-base16-light span.cm-def { color: #d28445; }
.cm-s-base16-light span.cm-bracket { color: #202020; }
.cm-s-base16-light span.cm-tag { color: #ac4142; }
.cm-s-base16-light span.cm-link { color: #aa759f; }
.cm-s-base16-light span.cm-error { background: #ac4142; color: #505050; }

.cm-s-base16-light .CodeMirror-activeline-background { background: #DDDCDC; }
.cm-s-base16-light .CodeMirror-matchingbracket { color: #f5f5f5 !important; background-color: #6A9FB5 !important}

/*

    Name:       Base16 Default Dark
    Author:     Chris Kempson (http://chriskempson.com)

    CodeMirror template by Jan T. Sott (https://github.com/idleberg/base16-codemirror)
    Original Base16 color scheme by Chris Kempson (https://github.com/chriskempson/base16)

*/

.cm-s-base16-dark.CodeMirror { background: #151515; color: #e0e0e0; }
.cm-s-base16-dark div.CodeMirror-selected { background: #303030; }
.cm-s-base16-dark .CodeMirror-line::selection, .cm-s-base16-dark .CodeMirror-line > span::selection, .cm-s-base16-dark .CodeMirror-line > span > span::selection { background: rgba(48, 48, 48, .99); }
.cm-s-base16-dark .CodeMirror-line::-moz-selection, .cm-s-base16-dark .CodeMirror-line > span::-moz-selection, .cm-s-base16-dark .CodeMirror-line > span > span::-moz-selection { background: rgba(48, 48, 48, .99); }
.cm-s-base16-dark .CodeMirror-gutters { background: #151515; border-right: 0px; }
.cm-s-base16-dark .CodeMirror-guttermarker { color: #ac4142; }
.cm-s-base16-dark .CodeMirror-guttermarker-subtle { color: #505050; }
.cm-s-base16-dark .CodeMirror-linenumber { color: #505050; }
.cm-s-base16-dark .CodeMirror-cursor { border-left: 1px solid #b0b0b0; }

.cm-s-base16-dark span.cm-comment { color: #8f5536; }
.cm-s-base16-dark span.cm-atom { color: #aa759f; }
.cm-s-base16-dark span.cm-number { color: #aa759f; }

.cm-s-base16-dark span.cm-property, .cm-s-base16-dark span.cm-attribute { color: #90a959; }
.cm-s-base16-dark span.cm-keyword { color: #ac4142; }
.cm-s-base16-dark span.cm-string { color: #f4bf75; }

.cm-s-base16-dark span.cm-variable { color: #90a959; }
.cm-s-base16-dark span.cm-variable-2 { color: #6a9fb5; }
.cm-s-base16-dark span.cm-def { color: #d28445; }
.cm-s-base16-dark span.cm-bracket { color: #e0e0e0; }
.cm-s-base16-dark span.cm-tag { color: #ac4142; }
.cm-s-base16-dark span.cm-link { color: #aa759f; }
.cm-s-base16-dark span.cm-error { background: #ac4142; color: #b0b0b0; }

.cm-s-base16-dark .CodeMirror-activeline-background { background: #202020; }
.cm-s-base16-dark .CodeMirror-matchingbracket { text-decoration: underline; color: white !important; }

/*

    Name:       dracula
    Author:     Michael Kaminsky (http://github.com/mkaminsky11)

    Original dracula color scheme by Zeno Rocha (https://github.com/zenorocha/dracula-theme)

*/


.cm-s-dracula.CodeMirror, .cm-s-dracula .CodeMirror-gutters {
  background-color: #282a36 !important;
  color: #f8f8f2 !important;
  border: none;
}
.cm-s-dracula .CodeMirror-gutters { color: #282a36; }
.cm-s-dracula .CodeMirror-cursor { border-left: solid thin #f8f8f0; }
.cm-s-dracula .CodeMirror-linenumber { color: #6D8A88; }
.cm-s-dracula .CodeMirror-selected { background: rgba(255, 255, 255, 0.10); }
.cm-s-dracula .CodeMirror-line::selection, .cm-s-dracula .CodeMirror-line > span::selection, .cm-s-dracula .CodeMirror-line > span > span::selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-dracula .CodeMirror-line::-moz-selection, .cm-s-dracula .CodeMirror-line > span::-moz-selection, .cm-s-dracula .CodeMirror-line > span > span::-moz-selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-dracula span.cm-comment { color: #6272a4; }
.cm-s-dracula span.cm-string, .cm-s-dracula span.cm-string-2 { color: #f1fa8c; }
.cm-s-dracula span.cm-number { color: #bd93f9; }
.cm-s-dracula span.cm-variable { color: #50fa7b; }
.cm-s-dracula span.cm-variable-2 { color: white; }
.cm-s-dracula span.cm-def { color: #50fa7b; }
.cm-s-dracula span.cm-operator { color: #ff79c6; }
.cm-s-dracula span.cm-keyword { color: #ff79c6; }
.cm-s-dracula span.cm-atom { color: #bd93f9; }
.cm-s-dracula span.cm-meta { color: #f8f8f2; }
.cm-s-dracula span.cm-tag { color: #ff79c6; }
.cm-s-dracula span.cm-attribute { color: #50fa7b; }
.cm-s-dracula span.cm-qualifier { color: #50fa7b; }
.cm-s-dracula span.cm-property { color: #66d9ef; }
.cm-s-dracula span.cm-builtin { color: #50fa7b; }
.cm-s-dracula span.cm-variable-3, .cm-s-dracula span.cm-type { color: #ffb86c; }

.cm-s-dracula .CodeMirror-activeline-background { background: rgba(255,255,255,0.1); }
.cm-s-dracula .CodeMirror-matchingbracket { text-decoration: underline; color: white !important; }

/*

    Name:       Hopscotch
    Author:     Jan T. Sott

    CodeMirror template by Jan T. Sott (https://github.com/idleberg/base16-codemirror)
    Original Base16 color scheme by Chris Kempson (https://github.com/chriskempson/base16)

*/

.cm-s-hopscotch.CodeMirror {background: #322931; color: #d5d3d5;}
.cm-s-hopscotch div.CodeMirror-selected {background: #433b42 !important;}
.cm-s-hopscotch .CodeMirror-gutters {background: #322931; border-right: 0px;}
.cm-s-hopscotch .CodeMirror-linenumber {color: #797379;}
.cm-s-hopscotch .CodeMirror-cursor {border-left: 1px solid #989498 !important;}

.cm-s-hopscotch span.cm-comment {color: #b33508;}
.cm-s-hopscotch span.cm-atom {color: #c85e7c;}
.cm-s-hopscotch span.cm-number {color: #c85e7c;}

.cm-s-hopscotch span.cm-property, .cm-s-hopscotch span.cm-attribute {color: #8fc13e;}
.cm-s-hopscotch span.cm-keyword {color: #dd464c;}
.cm-s-hopscotch span.cm-string {color: #fdcc59;}

.cm-s-hopscotch span.cm-variable {color: #8fc13e;}
.cm-s-hopscotch span.cm-variable-2 {color: #1290bf;}
.cm-s-hopscotch span.cm-def {color: #fd8b19;}
.cm-s-hopscotch span.cm-error {background: #dd464c; color: #989498;}
.cm-s-hopscotch span.cm-bracket {color: #d5d3d5;}
.cm-s-hopscotch span.cm-tag {color: #dd464c;}
.cm-s-hopscotch span.cm-link {color: #c85e7c;}

.cm-s-hopscotch .CodeMirror-matchingbracket { text-decoration: underline; color: white !important;}
.cm-s-hopscotch .CodeMirror-activeline-background { background: #302020; }

/****************************************************************/
/*   Based on mbonaci's Brackets mbo theme                      */
/*   https://github.com/mbonaci/global/blob/master/Mbo.tmTheme  */
/*   Create your own: http://tmtheme-editor.herokuapp.com       */
/****************************************************************/

.cm-s-mbo.CodeMirror { background: #2c2c2c; color: #ffffec; }
.cm-s-mbo div.CodeMirror-selected { background: #716C62; }
.cm-s-mbo .CodeMirror-line::selection, .cm-s-mbo .CodeMirror-line > span::selection, .cm-s-mbo .CodeMirror-line > span > span::selection { background: rgba(113, 108, 98, .99); }
.cm-s-mbo .CodeMirror-line::-moz-selection, .cm-s-mbo .CodeMirror-line > span::-moz-selection, .cm-s-mbo .CodeMirror-line > span > span::-moz-selection { background: rgba(113, 108, 98, .99); }
.cm-s-mbo .CodeMirror-gutters { background: #4e4e4e; border-right: 0px; }
.cm-s-mbo .CodeMirror-guttermarker { color: white; }
.cm-s-mbo .CodeMirror-guttermarker-subtle { color: grey; }
.cm-s-mbo .CodeMirror-linenumber { color: #dadada; }
.cm-s-mbo .CodeMirror-cursor { border-left: 1px solid #ffffec; }

.cm-s-mbo span.cm-comment { color: #95958a; }
.cm-s-mbo span.cm-atom { color: #00a8c6; }
.cm-s-mbo span.cm-number { color: #00a8c6; }

.cm-s-mbo span.cm-property, .cm-s-mbo span.cm-attribute { color: #9ddfe9; }
.cm-s-mbo span.cm-keyword { color: #ffb928; }
.cm-s-mbo span.cm-string { color: #ffcf6c; }
.cm-s-mbo span.cm-string.cm-property { color: #ffffec; }

.cm-s-mbo span.cm-variable { color: #ffffec; }
.cm-s-mbo span.cm-variable-2 { color: #00a8c6; }
.cm-s-mbo span.cm-def { color: #ffffec; }
.cm-s-mbo span.cm-bracket { color: #fffffc; font-weight: bold; }
.cm-s-mbo span.cm-tag { color: #9ddfe9; }
.cm-s-mbo span.cm-link { color: #f54b07; }
.cm-s-mbo span.cm-error { border-bottom: #636363; color: #ffffec; }
.cm-s-mbo span.cm-qualifier { color: #ffffec; }

.cm-s-mbo .CodeMirror-activeline-background { background: #494b41; }
.cm-s-mbo .CodeMirror-matchingbracket { color: #ffb928 !important; }
.cm-s-mbo .CodeMirror-matchingtag { background: rgba(255, 255, 255, .37); }

/*
  MDN-LIKE Theme - Mozilla
  Ported to CodeMirror by Peter Kroon <plakroon@gmail.com>
  Report bugs/issues here: https://github.com/codemirror/CodeMirror/issues
  GitHub: @peterkroon

  The mdn-like theme is inspired on the displayed code examples at: https://developer.mozilla.org/en-US/docs/Web/CSS/animation

*/
.cm-s-mdn-like.CodeMirror { color: #999; background-color: #fff; }
.cm-s-mdn-like div.CodeMirror-selected { background: #cfc; }
.cm-s-mdn-like .CodeMirror-line::selection, .cm-s-mdn-like .CodeMirror-line > span::selection, .cm-s-mdn-like .CodeMirror-line > span > span::selection { background: #cfc; }
.cm-s-mdn-like .CodeMirror-line::-moz-selection, .cm-s-mdn-like .CodeMirror-line > span::-moz-selection, .cm-s-mdn-like .CodeMirror-line > span > span::-moz-selection { background: #cfc; }

.cm-s-mdn-like .CodeMirror-gutters { background: #f8f8f8; border-left: 6px solid rgba(0,83,159,0.65); color: #333; }
.cm-s-mdn-like .CodeMirror-linenumber { color: #aaa; padding-left: 8px; }
.cm-s-mdn-like .CodeMirror-cursor { border-left: 2px solid #222; }

.cm-s-mdn-like .cm-keyword { color: #6262FF; }
.cm-s-mdn-like .cm-atom { color: #F90; }
.cm-s-mdn-like .cm-number { color:  #ca7841; }
.cm-s-mdn-like .cm-def { color: #8DA6CE; }
.cm-s-mdn-like span.cm-variable-2, .cm-s-mdn-like span.cm-tag { color: #690; }
.cm-s-mdn-like span.cm-variable-3, .cm-s-mdn-like span.cm-def, .cm-s-mdn-like span.cm-type { color: #07a; }

.cm-s-mdn-like .cm-variable { color: #07a; }
.cm-s-mdn-like .cm-property { color: #905; }
.cm-s-mdn-like .cm-qualifier { color: #690; }

.cm-s-mdn-like .cm-operator { color: #cda869; }
.cm-s-mdn-like .cm-comment { color:#777; font-weight:normal; }
.cm-s-mdn-like .cm-string { color:#07a; font-style:italic; }
.cm-s-mdn-like .cm-string-2 { color:#bd6b18; } /*?*/
.cm-s-mdn-like .cm-meta { color: #000; } /*?*/
.cm-s-mdn-like .cm-builtin { color: #9B7536; } /*?*/
.cm-s-mdn-like .cm-tag { color: #997643; }
.cm-s-mdn-like .cm-attribute { color: #d6bb6d; } /*?*/
.cm-s-mdn-like .cm-header { color: #FF6400; }
.cm-s-mdn-like .cm-hr { color: #AEAEAE; }
.cm-s-mdn-like .cm-link { color:#ad9361; font-style:italic; text-decoration:none; }
.cm-s-mdn-like .cm-error { border-bottom: 1px solid red; }

div.cm-s-mdn-like .CodeMirror-activeline-background { background: #efefff; }
div.cm-s-mdn-like span.CodeMirror-matchingbracket { outline:1px solid grey; color: inherit; }

.cm-s-mdn-like.CodeMirror { background-image: url(data:image/png;base64,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); }

/*

    Name:       seti
    Author:     Michael Kaminsky (http://github.com/mkaminsky11)

    Original seti color scheme by Jesse Weed (https://github.com/jesseweed/seti-syntax)

*/


.cm-s-seti.CodeMirror {
  background-color: #151718 !important;
  color: #CFD2D1 !important;
  border: none;
}
.cm-s-seti .CodeMirror-gutters {
  color: #404b53;
  background-color: #0E1112;
  border: none;
}
.cm-s-seti .CodeMirror-cursor { border-left: solid thin #f8f8f0; }
.cm-s-seti .CodeMirror-linenumber { color: #6D8A88; }
.cm-s-seti.CodeMirror-focused div.CodeMirror-selected { background: rgba(255, 255, 255, 0.10); }
.cm-s-seti .CodeMirror-line::selection, .cm-s-seti .CodeMirror-line > span::selection, .cm-s-seti .CodeMirror-line > span > span::selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-seti .CodeMirror-line::-moz-selection, .cm-s-seti .CodeMirror-line > span::-moz-selection, .cm-s-seti .CodeMirror-line > span > span::-moz-selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-seti span.cm-comment { color: #41535b; }
.cm-s-seti span.cm-string, .cm-s-seti span.cm-string-2 { color: #55b5db; }
.cm-s-seti span.cm-number { color: #cd3f45; }
.cm-s-seti span.cm-variable { color: #55b5db; }
.cm-s-seti span.cm-variable-2 { color: #a074c4; }
.cm-s-seti span.cm-def { color: #55b5db; }
.cm-s-seti span.cm-keyword { color: #ff79c6; }
.cm-s-seti span.cm-operator { color: #9fca56; }
.cm-s-seti span.cm-keyword { color: #e6cd69; }
.cm-s-seti span.cm-atom { color: #cd3f45; }
.cm-s-seti span.cm-meta { color: #55b5db; }
.cm-s-seti span.cm-tag { color: #55b5db; }
.cm-s-seti span.cm-attribute { color: #9fca56; }
.cm-s-seti span.cm-qualifier { color: #9fca56; }
.cm-s-seti span.cm-property { color: #a074c4; }
.cm-s-seti span.cm-variable-3, .cm-s-seti span.cm-type { color: #9fca56; }
.cm-s-seti span.cm-builtin { color: #9fca56; }
.cm-s-seti .CodeMirror-activeline-background { background: #101213; }
.cm-s-seti .CodeMirror-matchingbracket { text-decoration: underline; color: white !important; }

/*
Solarized theme for code-mirror
http://ethanschoonover.com/solarized
*/

/*
Solarized color palette
http://ethanschoonover.com/solarized/img/solarized-palette.png
*/

.solarized.base03 { color: #002b36; }
.solarized.base02 { color: #073642; }
.solarized.base01 { color: #586e75; }
.solarized.base00 { color: #657b83; }
.solarized.base0 { color: #839496; }
.solarized.base1 { color: #93a1a1; }
.solarized.base2 { color: #eee8d5; }
.solarized.base3  { color: #fdf6e3; }
.solarized.solar-yellow  { color: #b58900; }
.solarized.solar-orange  { color: #cb4b16; }
.solarized.solar-red { color: #dc322f; }
.solarized.solar-magenta { color: #d33682; }
.solarized.solar-violet  { color: #6c71c4; }
.solarized.solar-blue { color: #268bd2; }
.solarized.solar-cyan { color: #2aa198; }
.solarized.solar-green { color: #859900; }

/* Color scheme for code-mirror */

.cm-s-solarized {
  line-height: 1.45em;
  color-profile: sRGB;
  rendering-intent: auto;
}
.cm-s-solarized.cm-s-dark {
  color: #839496;
  background-color: #002b36;
  text-shadow: #002b36 0 1px;
}
.cm-s-solarized.cm-s-light {
  background-color: #fdf6e3;
  color: #657b83;
  text-shadow: #eee8d5 0 1px;
}

.cm-s-solarized .CodeMirror-widget {
  text-shadow: none;
}

.cm-s-solarized .cm-header { color: #586e75; }
.cm-s-solarized .cm-quote { color: #93a1a1; }

.cm-s-solarized .cm-keyword { color: #cb4b16; }
.cm-s-solarized .cm-atom { color: #d33682; }
.cm-s-solarized .cm-number { color: #d33682; }
.cm-s-solarized .cm-def { color: #2aa198; }

.cm-s-solarized .cm-variable { color: #839496; }
.cm-s-solarized .cm-variable-2 { color: #b58900; }
.cm-s-solarized .cm-variable-3, .cm-s-solarized .cm-type { color: #6c71c4; }

.cm-s-solarized .cm-property { color: #2aa198; }
.cm-s-solarized .cm-operator { color: #6c71c4; }

.cm-s-solarized .cm-comment { color: #586e75; font-style:italic; }

.cm-s-solarized .cm-string { color: #859900; }
.cm-s-solarized .cm-string-2 { color: #b58900; }

.cm-s-solarized .cm-meta { color: #859900; }
.cm-s-solarized .cm-qualifier { color: #b58900; }
.cm-s-solarized .cm-builtin { color: #d33682; }
.cm-s-solarized .cm-bracket { color: #cb4b16; }
.cm-s-solarized .CodeMirror-matchingbracket { color: #859900; }
.cm-s-solarized .CodeMirror-nonmatchingbracket { color: #dc322f; }
.cm-s-solarized .cm-tag { color: #93a1a1; }
.cm-s-solarized .cm-attribute { color: #2aa198; }
.cm-s-solarized .cm-hr {
  color: transparent;
  border-top: 1px solid #586e75;
  display: block;
}
.cm-s-solarized .cm-link { color: #93a1a1; cursor: pointer; }
.cm-s-solarized .cm-special { color: #6c71c4; }
.cm-s-solarized .cm-em {
  color: #999;
  text-decoration: underline;
  text-decoration-style: dotted;
}
.cm-s-solarized .cm-error,
.cm-s-solarized .cm-invalidchar {
  color: #586e75;
  border-bottom: 1px dotted #dc322f;
}

.cm-s-solarized.cm-s-dark div.CodeMirror-selected { background: #073642; }
.cm-s-solarized.cm-s-dark.CodeMirror ::selection { background: rgba(7, 54, 66, 0.99); }
.cm-s-solarized.cm-s-dark .CodeMirror-line::-moz-selection, .cm-s-dark .CodeMirror-line > span::-moz-selection, .cm-s-dark .CodeMirror-line > span > span::-moz-selection { background: rgba(7, 54, 66, 0.99); }

.cm-s-solarized.cm-s-light div.CodeMirror-selected { background: #eee8d5; }
.cm-s-solarized.cm-s-light .CodeMirror-line::selection, .cm-s-light .CodeMirror-line > span::selection, .cm-s-light .CodeMirror-line > span > span::selection { background: #eee8d5; }
.cm-s-solarized.cm-s-light .CodeMirror-line::-moz-selection, .cm-s-ligh .CodeMirror-line > span::-moz-selection, .cm-s-ligh .CodeMirror-line > span > span::-moz-selection { background: #eee8d5; }

/* Editor styling */



/* Little shadow on the view-port of the buffer view */
.cm-s-solarized.CodeMirror {
  -moz-box-shadow: inset 7px 0 12px -6px #000;
  -webkit-box-shadow: inset 7px 0 12px -6px #000;
  box-shadow: inset 7px 0 12px -6px #000;
}

/* Remove gutter border */
.cm-s-solarized .CodeMirror-gutters {
  border-right: 0;
}

/* Gutter colors and line number styling based of color scheme (dark / light) */

/* Dark */
.cm-s-solarized.cm-s-dark .CodeMirror-gutters {
  background-color: #073642;
}

.cm-s-solarized.cm-s-dark .CodeMirror-linenumber {
  color: #586e75;
  text-shadow: #021014 0 -1px;
}

/* Light */
.cm-s-solarized.cm-s-light .CodeMirror-gutters {
  background-color: #eee8d5;
}

.cm-s-solarized.cm-s-light .CodeMirror-linenumber {
  color: #839496;
}

/* Common */
.cm-s-solarized .CodeMirror-linenumber {
  padding: 0 5px;
}
.cm-s-solarized .CodeMirror-guttermarker-subtle { color: #586e75; }
.cm-s-solarized.cm-s-dark .CodeMirror-guttermarker { color: #ddd; }
.cm-s-solarized.cm-s-light .CodeMirror-guttermarker { color: #cb4b16; }

.cm-s-solarized .CodeMirror-gutter .CodeMirror-gutter-text {
  color: #586e75;
}

/* Cursor */
.cm-s-solarized .CodeMirror-cursor { border-left: 1px solid #819090; }

/* Fat cursor */
.cm-s-solarized.cm-s-light.cm-fat-cursor .CodeMirror-cursor { background: #77ee77; }
.cm-s-solarized.cm-s-light .cm-animate-fat-cursor { background-color: #77ee77; }
.cm-s-solarized.cm-s-dark.cm-fat-cursor .CodeMirror-cursor { background: #586e75; }
.cm-s-solarized.cm-s-dark .cm-animate-fat-cursor { background-color: #586e75; }

/* Active line */
.cm-s-solarized.cm-s-dark .CodeMirror-activeline-background {
  background: rgba(255, 255, 255, 0.06);
}
.cm-s-solarized.cm-s-light .CodeMirror-activeline-background {
  background: rgba(0, 0, 0, 0.06);
}

.cm-s-the-matrix.CodeMirror { background: #000000; color: #00FF00; }
.cm-s-the-matrix div.CodeMirror-selected { background: #2D2D2D; }
.cm-s-the-matrix .CodeMirror-line::selection, .cm-s-the-matrix .CodeMirror-line > span::selection, .cm-s-the-matrix .CodeMirror-line > span > span::selection { background: rgba(45, 45, 45, 0.99); }
.cm-s-the-matrix .CodeMirror-line::-moz-selection, .cm-s-the-matrix .CodeMirror-line > span::-moz-selection, .cm-s-the-matrix .CodeMirror-line > span > span::-moz-selection { background: rgba(45, 45, 45, 0.99); }
.cm-s-the-matrix .CodeMirror-gutters { background: #060; border-right: 2px solid #00FF00; }
.cm-s-the-matrix .CodeMirror-guttermarker { color: #0f0; }
.cm-s-the-matrix .CodeMirror-guttermarker-subtle { color: white; }
.cm-s-the-matrix .CodeMirror-linenumber { color: #FFFFFF; }
.cm-s-the-matrix .CodeMirror-cursor { border-left: 1px solid #00FF00; }

.cm-s-the-matrix span.cm-keyword { color: #008803; font-weight: bold; }
.cm-s-the-matrix span.cm-atom { color: #3FF; }
.cm-s-the-matrix span.cm-number { color: #FFB94F; }
.cm-s-the-matrix span.cm-def { color: #99C; }
.cm-s-the-matrix span.cm-variable { color: #F6C; }
.cm-s-the-matrix span.cm-variable-2 { color: #C6F; }
.cm-s-the-matrix span.cm-variable-3, .cm-s-the-matrix span.cm-type { color: #96F; }
.cm-s-the-matrix span.cm-property { color: #62FFA0; }
.cm-s-the-matrix span.cm-operator { color: #999; }
.cm-s-the-matrix span.cm-comment { color: #CCCCCC; }
.cm-s-the-matrix span.cm-string { color: #39C; }
.cm-s-the-matrix span.cm-meta { color: #C9F; }
.cm-s-the-matrix span.cm-qualifier { color: #FFF700; }
.cm-s-the-matrix span.cm-builtin { color: #30a; }
.cm-s-the-matrix span.cm-bracket { color: #cc7; }
.cm-s-the-matrix span.cm-tag { color: #FFBD40; }
.cm-s-the-matrix span.cm-attribute { color: #FFF700; }
.cm-s-the-matrix span.cm-error { color: #FF0000; }

.cm-s-the-matrix .CodeMirror-activeline-background { background: #040; }

/*
Copyright (C) 2011 by MarkLogic Corporation
Author: Mike Brevoort <mike@brevoort.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
.cm-s-xq-light span.cm-keyword { line-height: 1em; font-weight: bold; color: #5A5CAD; }
.cm-s-xq-light span.cm-atom { color: #6C8CD5; }
.cm-s-xq-light span.cm-number { color: #164; }
.cm-s-xq-light span.cm-def { text-decoration:underline; }
.cm-s-xq-light span.cm-variable { color: black; }
.cm-s-xq-light span.cm-variable-2 { color:black; }
.cm-s-xq-light span.cm-variable-3, .cm-s-xq-light span.cm-type { color: black; }
.cm-s-xq-light span.cm-property {}
.cm-s-xq-light span.cm-operator {}
.cm-s-xq-light span.cm-comment { color: #0080FF; font-style: italic; }
.cm-s-xq-light span.cm-string { color: red; }
.cm-s-xq-light span.cm-meta { color: yellow; }
.cm-s-xq-light span.cm-qualifier { color: grey; }
.cm-s-xq-light span.cm-builtin { color: #7EA656; }
.cm-s-xq-light span.cm-bracket { color: #cc7; }
.cm-s-xq-light span.cm-tag { color: #3F7F7F; }
.cm-s-xq-light span.cm-attribute { color: #7F007F; }
.cm-s-xq-light span.cm-error { color: #f00; }

.cm-s-xq-light .CodeMirror-activeline-background { background: #e8f2ff; }
.cm-s-xq-light .CodeMirror-matchingbracket { outline:1px solid grey;color:black !important;background:yellow; }

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.CodeMirror {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;
  /* Changed to auto to autogrow */
}

.CodeMirror pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* This causes https://github.com/jupyter/jupyterlab/issues/522 */
/* May not cause it not because we changed it! */
.CodeMirror-lines {
  padding: var(--jp-code-padding) 0;
}

.CodeMirror-linenumber {
  padding: 0 8px;
}

.jp-CodeMirrorEditor-static {
  margin: var(--jp-code-padding);
}

.jp-CodeMirrorEditor,
.jp-CodeMirrorEditor-static {
  cursor: text;
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
  display: none;
}

.CodeMirror-gutters {
  border-right: 1px solid var(--jp-border-color2);
  background-color: var(--jp-layout-color0);
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.CodeMirror-selectedtext.cm-searching {
  background-color: var(--jp-search-selected-match-background-color) !important;
  color: var(--jp-search-selected-match-color) !important;
}

.cm-searching {
  background-color: var(
    --jp-search-unselected-match-background-color
  ) !important;
  color: var(--jp-search-unselected-match-color) !important;
}

.CodeMirror-focused .CodeMirror-selected {
  background-color: var(--jp-editor-selected-focused-background);
}

.CodeMirror-selected {
  background-color: var(--jp-editor-selected-background);
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/**
 * Here is our jupyter theme for CodeMirror syntax highlighting
 * This is used in our marked.js syntax highlighting and CodeMirror itself
 * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
 * This came from the classic notebook, which came form highlight.js/GitHub
 */

/**
 * CodeMirror themes are handling the background/color in this way. This works
 * fine for CodeMirror editors outside the notebook, but the notebook styles
 * these things differently.
 */
.CodeMirror.cm-s-jupyter {
  background: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* In the notebook, we want this styling to be handled by its container */
.jp-CodeConsole .CodeMirror.cm-s-jupyter,
.jp-Notebook .CodeMirror.cm-s-jupyter {
  background: transparent;
}

.cm-s-jupyter .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}
.cm-s-jupyter span.cm-keyword {
  color: var(--jp-mirror-editor-keyword-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-atom {
  color: var(--jp-mirror-editor-atom-color);
}
.cm-s-jupyter span.cm-number {
  color: var(--jp-mirror-editor-number-color);
}
.cm-s-jupyter span.cm-def {
  color: var(--jp-mirror-editor-def-color);
}
.cm-s-jupyter span.cm-variable {
  color: var(--jp-mirror-editor-variable-color);
}
.cm-s-jupyter span.cm-variable-2 {
  color: var(--jp-mirror-editor-variable-2-color);
}
.cm-s-jupyter span.cm-variable-3 {
  color: var(--jp-mirror-editor-variable-3-color);
}
.cm-s-jupyter span.cm-punctuation {
  color: var(--jp-mirror-editor-punctuation-color);
}
.cm-s-jupyter span.cm-property {
  color: var(--jp-mirror-editor-property-color);
}
.cm-s-jupyter span.cm-operator {
  color: var(--jp-mirror-editor-operator-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-comment {
  color: var(--jp-mirror-editor-comment-color);
  font-style: italic;
}
.cm-s-jupyter span.cm-string {
  color: var(--jp-mirror-editor-string-color);
}
.cm-s-jupyter span.cm-string-2 {
  color: var(--jp-mirror-editor-string-2-color);
}
.cm-s-jupyter span.cm-meta {
  color: var(--jp-mirror-editor-meta-color);
}
.cm-s-jupyter span.cm-qualifier {
  color: var(--jp-mirror-editor-qualifier-color);
}
.cm-s-jupyter span.cm-builtin {
  color: var(--jp-mirror-editor-builtin-color);
}
.cm-s-jupyter span.cm-bracket {
  color: var(--jp-mirror-editor-bracket-color);
}
.cm-s-jupyter span.cm-tag {
  color: var(--jp-mirror-editor-tag-color);
}
.cm-s-jupyter span.cm-attribute {
  color: var(--jp-mirror-editor-attribute-color);
}
.cm-s-jupyter span.cm-header {
  color: var(--jp-mirror-editor-header-color);
}
.cm-s-jupyter span.cm-quote {
  color: var(--jp-mirror-editor-quote-color);
}
.cm-s-jupyter span.cm-link {
  color: var(--jp-mirror-editor-link-color);
}
.cm-s-jupyter span.cm-error {
  color: var(--jp-mirror-editor-error-color);
}
.cm-s-jupyter span.cm-hr {
  color: #999;
}

.cm-s-jupyter span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}

.cm-s-jupyter .CodeMirror-activeline-background,
.cm-s-jupyter .CodeMirror-gutter {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0px;
  padding: 0px;
  line-height: normal;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}
.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}
.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
}
.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
}
.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}
.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}
.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0em;
}

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: 12px;
  table-layout: fixed;
  margin-left: auto;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon table {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0px;
}

.jp-RenderedHTMLCommon p {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}
/* ...or leave it untouched if they don't */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}
.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}
.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}
.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}
.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}
.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}
.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}
.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}
.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: 0.8em;
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  border-bottom: none;
  height: auto;
  margin: var(--jp-toolbar-header-margin);
  box-shadow: none;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 4px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0px 2px;
  padding: 0px 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0px;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar.jp-Toolbar {
  padding: 0px;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  justify-content: space-evenly;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
  flex: 1;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: 100%;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px 12px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-item.jp-mod-selected {
  color: white;
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
  color: limegreen;
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0px;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-DirListing-deadSpace {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

.jp-FileDialog.jp-mod-conflict input {
  color: red;
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
}

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
}

.jp-OutputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-output {
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0px;
  padding: 0px;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0px;
  flex: 1 1 auto;
}

.jp-OutputArea-executeResult.jp-RenderedText {
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-OutputArea-stdin {
  line-height: var(--jp-code-line-height);
  padding-top: var(--jp-code-padding);
  display: flex;
}

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0px;
  bottom: 0px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0px;
  width: 100%;
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
}

.jp-InputArea-editor {
  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0px;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  flex: 1 1 auto;
}

.jp-Placeholder-prompt {
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  border: none;
  background: transparent;
  height: 20px;
  box-sizing: border-box;
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0px;
  margin: 0px;
  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 200px;
  box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
  margin-left: var(--jp-private-cell-scrolling-output-offset);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: 2px;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: flex;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0px;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-NotebookTools-tool {
  padding: 0px 12px 0 12px;
}

.jp-ActiveCellTool {
  padding: 12px;
  background-color: var(--jp-layout-color1);
  border-top: none !important;
}

.jp-ActiveCellTool .jp-InputArea-prompt {
  flex: 0 0 auto;
  padding-left: 0px;
}

.jp-ActiveCellTool .jp-InputArea-editor {
  flex: 1 1 auto;
  background: var(--jp-cell-editor-background);
  border-color: var(--jp-cell-editor-border-color);
}

.jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
  background: transparent;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0px 12px 0px;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label {
  line-height: 1.4;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

</style>

    <style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
    0px 1px 1px 0px var(--jp-shadow-penumbra-color),
    0px 1px 3px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
    0px 2px 2px 0px var(--jp-shadow-penumbra-color),
    0px 1px 5px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
    0px 4px 5px 0px var(--jp-shadow-penumbra-color),
    0px 1px 10px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
    0px 6px 10px 0px var(--jp-shadow-penumbra-color),
    0px 1px 18px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
    0px 8px 10px 1px var(--jp-shadow-penumbra-color),
    0px 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
    0px 12px 17px 2px var(--jp-shadow-penumbra-color),
    0px 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
    0px 16px 24px 2px var(--jp-shadow-penumbra-color),
    0px 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
    0px 20px 31px 3px var(--jp-shadow-penumbra-color),
    0px 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
    0px 24px 38px 3px var(--jp-shadow-penumbra-color),
    0px 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;

  --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
    Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;

  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);

  --jp-content-link-color: var(--md-blue-700);

  --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-700);
  --jp-brand-color1: var(--md-blue-500);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-700);
  --jp-accent-color1: var(--md-green-500);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-700);
  --jp-warn-color1: var(--md-orange-500);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-700);
  --jp-error-color1: var(--md-red-500);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-700);
  --jp-success-color1: var(--md-green-500);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-700);
  --jp-info-color1: var(--md-cyan-500);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;

  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;

  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);

  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: 'Source Code Pro', monospace;
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-border-color1);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: #05a;
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #aa22ff;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #aa22ff;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 180px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);
}
</style>

<style type="text/css">
a.anchor-link {
   display: none;
}
.highlight  {
    margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
    overflow: hidden;
}

.jp-InputArea-editor {
    overflow: hidden;
}

@media print {
  body {
    margin: 0;
  }
}
</style>



<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                },
                "HTML-CSS": {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">


<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="A-Primer-on-Graph-Neural-Networks">A Primer on Graph Neural Networks<a class="anchor-link" href="#A-Primer-on-Graph-Neural-Networks">&#182;</a></h1><p><strong>Author</strong>: <a href="https://alessiodevoto.github.io/">Alessio Devoto</a></p>
<p>This is an introductory tutorial to <a href="https://pytorch-geometric.readthedocs.io/en/latest/">PyTorch Geometric</a> a library for the design of deep Graph Neural Networks(GNNs). The first part is a re-adaptation of the documentation from the PyG website, training a GNN for graph classification on the <a href="https://paperswithcode.com/dataset/mutag">MUTAG</a> dataset (using <a href="https://www.pytorchlightning.ai/">PyTorch Lightning</a> for the training loop). The notebook is inspired by <a href="https://sscardapane.it/">Simone Scardapane</a>'s material on GNNs.</p>

</div>
</div>

<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a href="https://colab.research.google.com/github/alessiodevoto/notebooks/blob/main/A_Primer_on_Graph_Neural_Networks_(Liverpool).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

</div>
</div>

<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="1.-&#128663;-Setup-the-colab-environment">1. &#128663; Setup the colab environment<a class="anchor-link" href="#1.-&#128663;-Setup-the-colab-environment">&#182;</a></h2>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># We use a cpu based installation for torch geometric</span>
<span class="c1"># More info here https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html</span>
<span class="err">!</span><span class="n">pip</span> <span class="n">install</span> <span class="n">torch_geometric</span>
<span class="err">!</span><span class="n">pip</span> <span class="n">install</span> <span class="n">pyg_lib</span> <span class="n">torch_scatter</span> <span class="n">torch_sparse</span> <span class="n">torch_cluster</span> <span class="n">torch_spline_conv</span> <span class="o">-</span><span class="n">f</span> <span class="n">https</span><span class="p">:</span><span class="o">//</span><span class="n">data</span><span class="o">.</span><span class="n">pyg</span><span class="o">.</span><span class="n">org</span><span class="o">/</span><span class="n">whl</span><span class="o">/</span><span class="n">torch</span><span class="o">-</span><span class="mf">2.0</span><span class="o">.</span><span class="mi">0</span><span class="o">+</span><span class="n">cpu</span><span class="o">.</span><span class="n">html</span>
<span class="err">!</span><span class="n">pip</span> <span class="n">install</span> <span class="n">pytorch</span><span class="o">-</span><span class="n">lightning</span> <span class="o">--</span><span class="n">quiet</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># PyTorch imports</span>
<span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">from</span> <span class="nn">torch.nn</span> <span class="kn">import</span> <span class="n">functional</span> <span class="k">as</span> <span class="n">F</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># PyTorch-related imports</span>
<span class="kn">import</span> <span class="nn">torch_geometric</span> <span class="k">as</span> <span class="nn">ptgeom</span>
<span class="kn">import</span> <span class="nn">torch_scatter</span><span class="o">,</span> <span class="nn">torch_sparse</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="kn">import</span> <span class="nn">pytorch_lightning</span> <span class="k">as</span> <span class="nn">ptlight</span>
<span class="kn">from</span> <span class="nn">pytorch_lightning.loggers</span> <span class="kn">import</span> <span class="n">TensorBoardLogger</span>
<span class="kn">from</span> <span class="nn">torchmetrics.functional</span> <span class="kn">import</span> <span class="n">accuracy</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Other imports</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">networkx</span> <span class="k">as</span> <span class="nn">nx</span>
<span class="kn">import</span> <span class="nn">matplotlib</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">matplotlib.colors</span> <span class="k">as</span> <span class="nn">mcolors</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">matplotlib</span><span class="o">.</span><span class="n">rcParams</span><span class="p">[</span><span class="s1">&#39;figure.dpi&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">120</span> <span class="c1"># I like higher resolution plots :)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="2.-&#128190;-Data">2. &#128190; Data<a class="anchor-link" href="#2.-&#128190;-Data">&#182;</a></h2>
</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="2.1-Download-&amp;-Explore-Dataset">2.1 Download &amp; Explore Dataset<a class="anchor-link" href="#2.1-Download-&amp;-Explore-Dataset">&#182;</a></h3><p>Pytorch Geometric provides a number of datasets to use off-the-shelf, for all graph related tasks (graph, node or edge level tasks). Find a complete list <a href="https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html">here</a>.</p>
<p>In this tutorial, we will use the <strong>MUTAG</strong> dataset. See the MUTAG page on <a href="https://paperswithcode.com/dataset/mutag">Papers With Code</a> and related papers for more information about the dataset. This is a toy version, so we do not care too much about the final performance.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Download the dataset</span>
<span class="n">mutag</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">TUDataset</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s1">&#39;.&#39;</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">&#39;MUTAG&#39;</span><span class="p">)</span> <span class="c1"># just like a plain Torch Dataset</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>Downloading https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip
Extracting ./MUTAG/MUTAG.zip
Processing...
Done!
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Useful info stored in the dataset class</span>

<span class="nb">print</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">mutag</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">mutag</span><span class="o">.</span><span class="n">num_classes</span><span class="p">)</span> <span class="c1"># Binary (graph-level) classification</span>
<span class="nb">print</span><span class="p">(</span><span class="n">mutag</span><span class="o">.</span><span class="n">num_features</span><span class="p">)</span> <span class="c1"># One-hot encoding for each node type (atom)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">mutag</span><span class="o">.</span><span class="n">num_edge_features</span><span class="p">)</span> <span class="c1"># One-hot encoding for the bond type (we will ignore this)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>188
2
7
4
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Each graph in the dataset is represented as an instance of the generic Data object:</span>
<span class="c1"># https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data</span>
<span class="n">mutag_0</span> <span class="o">=</span> <span class="n">mutag</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">mutag_0</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>What is the meaning of each of these fields ?</p>
<p><img src="https://raw.githubusercontent.com/alessiodevoto/gnns_xai_liverpool/main/images/simple_graph.png" alt=""></p>

</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>⚠ Other datasets (e.g. for node level tasks) only contain one single <em>huge</em> Data object.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># node features</span>
<span class="n">mutag_0</span><span class="o">.</span><span class="n">x</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([17, 7])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># graph class (remember we only have two classes as this is a binary classfication problem)</span>
<span class="n">mutag_0</span><span class="o">.</span><span class="n">y</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>tensor([1])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># the adjacency matrix stored in COO format</span>
<span class="n">mutag_0</span><span class="o">.</span><span class="n">edge_index</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([2, 38])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># let&#39;s take a look at the first 4 edges</span>
<span class="n">mutag_0</span><span class="o">.</span><span class="n">edge_index</span><span class="p">[:,</span> <span class="p">:</span><span class="mi">4</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>tensor([[0, 0, 1, 1],
        [1, 5, 0, 2]])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Inside utils (https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html)</span>
<span class="c1"># there are a number of useful tools, e.g., we can check that the graph is undirected (the adjacency matrix is symmetric)</span>
<span class="n">ptgeom</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">is_undirected</span><span class="p">(</span><span class="n">mutag_0</span><span class="o">.</span><span class="n">edge_index</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>True</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># are there self loops in this graph ?</span>
<span class="n">ptgeom</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">contains_self_loops</span><span class="p">(</span><span class="n">mutag_0</span><span class="o">.</span><span class="n">edge_index</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>False</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># any isolated components ?</span>
<span class="n">ptgeom</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">contains_isolated_nodes</span><span class="p">(</span><span class="n">mutag_0</span><span class="o">.</span><span class="n">edge_index</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>False</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="2.2-Data-Visualization">2.2 Data Visualization<a class="anchor-link" href="#2.2-Data-Visualization">&#182;</a></h3><p>As always, it is crucial to visualize (if possible) the data structures we are dealing with.</p>
<p>In the case of graphs, this can be prohibitive due to very high number of nodes. Luckily all our molecules are quite small.</p>
<p>Let's define a simple function to plot a graph, using <code>matplotlib</code> and the <code>networkx</code> package</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># This one is copy-pasted from: https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing</span>
<span class="kn">import</span> <span class="nn">networkx</span> <span class="k">as</span> <span class="nn">nx</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">torch_geometric.utils</span> <span class="kn">import</span> <span class="n">to_networkx</span>
<span class="kn">from</span> <span class="nn">matplotlib.pyplot</span> <span class="kn">import</span> <span class="n">figure</span>

<span class="c1"># transform the pytorch geometric graph into networkx format</span>
<span class="k">def</span> <span class="nf">to_molecule</span><span class="p">(</span><span class="n">data</span><span class="p">:</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">data</span><span class="o">.</span><span class="n">Data</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">nx</span><span class="o">.</span><span class="n">classes</span><span class="o">.</span><span class="n">digraph</span><span class="o">.</span><span class="n">DiGraph</span><span class="p">:</span>
    <span class="n">ATOM_MAP</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;C&#39;</span><span class="p">,</span> <span class="s1">&#39;O&#39;</span><span class="p">,</span> <span class="s1">&#39;Cl&#39;</span><span class="p">,</span> <span class="s1">&#39;H&#39;</span><span class="p">,</span> <span class="s1">&#39;N&#39;</span><span class="p">,</span> <span class="s1">&#39;F&#39;</span><span class="p">,</span>
                <span class="s1">&#39;Br&#39;</span><span class="p">,</span> <span class="s1">&#39;S&#39;</span><span class="p">,</span> <span class="s1">&#39;P&#39;</span><span class="p">,</span> <span class="s1">&#39;I&#39;</span><span class="p">,</span> <span class="s1">&#39;Na&#39;</span><span class="p">,</span> <span class="s1">&#39;K&#39;</span><span class="p">,</span> <span class="s1">&#39;Li&#39;</span><span class="p">,</span> <span class="s1">&#39;Ca&#39;</span><span class="p">]</span>
    <span class="n">g</span> <span class="o">=</span> <span class="n">to_networkx</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">node_attrs</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;x&#39;</span><span class="p">])</span>
    <span class="k">for</span> <span class="n">u</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">g</span><span class="o">.</span><span class="n">nodes</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="kc">True</span><span class="p">):</span>
        <span class="n">data</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">ATOM_MAP</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">&#39;x&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="mf">1.0</span><span class="p">)]</span>
        <span class="k">del</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;x&#39;</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">g</span>

<span class="c1"># plot the molecule</span>
<span class="k">def</span> <span class="nf">draw_molecule</span><span class="p">(</span><span class="n">g</span><span class="p">,</span> <span class="n">edge_mask</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">draw_edge_labels</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">draw_node_labels</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="n">figsize</span> <span class="ow">or</span> <span class="p">(</span><span class="mi">4</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span>

    <span class="c1"># check if it&#39;s been already converted to a nx graph</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="nb">isinstance</span><span class="p">(</span><span class="n">g</span><span class="p">,</span> <span class="n">nx</span><span class="o">.</span><span class="n">classes</span><span class="o">.</span><span class="n">digraph</span><span class="o">.</span><span class="n">DiGraph</span><span class="p">):</span>
      <span class="n">g</span> <span class="o">=</span> <span class="n">to_molecule</span><span class="p">(</span><span class="n">g</span><span class="p">)</span>

    <span class="n">g</span> <span class="o">=</span> <span class="n">g</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span><span class="o">.</span><span class="n">to_undirected</span><span class="p">()</span>
    <span class="n">node_labels</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">u</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">g</span><span class="o">.</span><span class="n">nodes</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="kc">True</span><span class="p">):</span>
        <span class="n">node_labels</span><span class="p">[</span><span class="n">u</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;name&#39;</span><span class="p">]</span>
    <span class="n">pos</span> <span class="o">=</span> <span class="n">nx</span><span class="o">.</span><span class="n">planar_layout</span><span class="p">(</span><span class="n">g</span><span class="p">)</span>
    <span class="n">pos</span> <span class="o">=</span> <span class="n">nx</span><span class="o">.</span><span class="n">spring_layout</span><span class="p">(</span><span class="n">g</span><span class="p">,</span> <span class="n">pos</span><span class="o">=</span><span class="n">pos</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">edge_mask</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
        <span class="n">edge_color</span> <span class="o">=</span> <span class="s1">&#39;black&#39;</span>
        <span class="n">widths</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">edge_color</span> <span class="o">=</span> <span class="p">[</span><span class="n">edge_mask</span><span class="p">[(</span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">)]</span> <span class="k">for</span> <span class="n">u</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">g</span><span class="o">.</span><span class="n">edges</span><span class="p">()]</span>
        <span class="n">widths</span> <span class="o">=</span> <span class="p">[</span><span class="n">x</span> <span class="o">*</span> <span class="mi">10</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">edge_color</span><span class="p">]</span>
    <span class="n">nx</span><span class="o">.</span><span class="n">draw</span><span class="p">(</span><span class="n">g</span><span class="p">,</span> <span class="n">pos</span><span class="o">=</span><span class="n">pos</span><span class="p">,</span> <span class="n">labels</span><span class="o">=</span><span class="n">node_labels</span> <span class="k">if</span> <span class="n">draw_node_labels</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="n">widths</span><span class="p">,</span>
            <span class="n">edge_color</span><span class="o">=</span><span class="n">edge_color</span><span class="p">,</span> <span class="n">edge_cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">Blues</span><span class="p">,</span>
            <span class="n">node_color</span><span class="o">=</span><span class="s1">&#39;azure&#39;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">draw_edge_labels</span> <span class="ow">and</span> <span class="n">edge_mask</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
        <span class="n">edge_labels</span> <span class="o">=</span> <span class="p">{</span><span class="n">k</span><span class="p">:</span> <span class="p">(</span><span class="s1">&#39;</span><span class="si">%.2f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="n">v</span><span class="p">)</span> <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">edge_mask</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>
        <span class="n">nx</span><span class="o">.</span><span class="n">draw_networkx_edge_labels</span><span class="p">(</span><span class="n">g</span><span class="p">,</span> <span class="n">pos</span><span class="p">,</span> <span class="n">edge_labels</span><span class="o">=</span><span class="n">edge_labels</span><span class="p">,</span>
                                    <span class="n">font_color</span><span class="o">=</span><span class="s1">&#39;red&#39;</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">ax</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
      <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

<span class="c1"># Let&#39;s try it</span>
<span class="n">draw_molecule</span><span class="p">(</span><span class="n">mutag_0</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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Xx0/TuYPfsCOjz7C4fB89YsuuqhWkKelpdG+fXsfrYVPPvmERx55hNWrV1c/Z7PZ+OarrzBcLmxB6rknBOUqIuFD4S4SJoJdc91ms3He+edTaRi1gv38888nLS2tVph37twZm836oPZHH33EI488wttvv139nN1uZ+rUqcyZM4c+ffpYmlvgD0loprzEHoW7SJgIVc31myZOZPiQIaSnpzN48GB69OjR6HvhW7duZf78+axdu7b6ubi4OKZNm8ZDDz3EpZdeWv18AuZIRSArx8WhXrvEJoW7SJgIVfW4e++/nxZNPMfmzZuZP38+f/vb36qfa9asGbfeeisPPvggPXr0qPM1NqAFgVnP7z5/czSJTmKTRqtEwkQk1lzPy8vjmmuu4YorrqgO9vj4eO644w6KiopYunRpvcHuFoe5RM3fARyqSnwi4ULhLhJBwqHmumEY5OTkcNVVVzFy5Eg2VrUlISGBu+++m+LiYp577jm6d+9u6XzugPfXP0bu8ynYJZZpKZxImLCysUsoa64bhsGGDRuYP38++VUT+wCSkpK44447mD17Np07d7ZwJg/nx7wH35RJdkmY99g1FC+xTj13kTBh5ZcxFDXXDcPg3XffZfjw4Vx33XXVwd6iRQtmzZrFwYMHeeqpp5oU7GAGciJmsZlErAd0Y79OJJppQp1ImIgDKiwcF6ya64ZhsGbNGubPn8/2Gh8iWrZsyT333MP999/PhRdeaOkaDWHn29K4lTUernOOiavxUKCL1KZheZEw4cScOW5VoGquu1wu3n77bebPn8/HH39c/Xzr1q35+c9/zsyZMy0VsBGR0FG4i4QJA7OufDB/IWvWXK+srGT16tU8+uij7Nq1q/qYNm3aMHPmTO69917atm0bxNaJSGNpWF4kTBiYw83BXO+eALgqK3nttdd49NFHKSgoqH6tbdu23H///cyYMYM2bdoEsVUi0lTquYuEmD9miTfuwgbr33iD3/z61+zbt6/66fbt2zNr1izuvvtuWrduHexWiYgfqOcuEkKVwBkCW4LVk6cWLWLegw9W//eFF17I7NmzufPOOz3u7iYikUE9d5EQqSRwpVe9MQyD7R9+yPevuAKXy8VFF13Er371K37605/SokVTC9GKSDhQuIuEQKiC3eVy8Z8vv+S6K66gorycBx54gB//+Mc0b948yC0RkUBSuIsE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
"
>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># @title Visualize some graphs { run: &quot;auto&quot; }</span>
<span class="n">mutag_idx</span> <span class="o">=</span> <span class="mi">35</span> <span class="c1"># @param {type:&quot;slider&quot;, min:0, max:187, step:1}</span>

<span class="n">draw_molecule</span><span class="p">(</span><span class="n">mutag</span><span class="p">[</span><span class="n">mutag_idx</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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6W41a9Zk+/btVKhQAYAhQ4awS0X+eeE5roxZqV6zptNtbdtcuXTJ4XbffvcdL730En/72994//33uexg+2C8OZfgrnOuJiqxZQ376vhxLly/zs85OZy8fJn127cz6C9/UT2S1LbynKurOIG1BX8Tzz/LysjIYNu2baQcOuR0W60fXUDwnny2efjhWEfVlyv0isx/PxT7x22zZs3YtGkTYWFhmM1mHn/8cY4dO+aFmouS+KK3zTYjB6zXmMqVK5Nx44bd640nb8799Tz213oJjdhSggYaC9buei0DvMVi4ejRo8yZM4fOnTtTsWJFevbsyZp//lPV57V+dOHvC0/4s06dOpGYmAjA9evX6dGjR8G638L/3J2/ZsLFCxecbmvbpmp0tN1tDAYD7dq35+effyYnJ4effvqJPo8/bnd7T9yc2/jreSwZ6oKA11KCKgo4WCve5d3h/gpMV65c4fPPP2fHjh3s2LGDn3766bZtjh09qnp/4ydP5rF+/QoWPFmdkFBkwZPR48fz5MCBqvblrxeFQDFgwAAuXrzI2LFjuXTpEj169CA5OZk777zT11UTxbTt2JF/797Nnp07GTxsmN3t8vLySN6zB4A2HTrY3c5gMBAZGUm5yMiC95ydT1qvRqi2XF+RVeGChNq1tj3FncUaXFmBKSsri+Tk5IJg/u2335a43T333MMjjzxCt27dePChhzBUqODV5/yympk2FEXh5ZdfZv78+QD84Q9/YMeOHURE+GtSUP25hXVwrCPpZ88SW78+RqORfUeOlDgVDiBx6VJG//Wv3NugAQdPnHD4WM+E9TGOjdpVCdWuRuhoVTgbfz6PJbgHCa2WQSyN4os1FL5bTt6zh/SzZ2kRG+twsQbbgijFKYpCamoqn332GTt27GDv3r1kZmbetl2FChXo3LkzjzzyCI888gh169Yt8nNny+JqLRz/zUkdaCwWC08++STr8zOb9e3bl7Vr15ZqvIdwndqGwxuTJzN3xgxi6tbloy1baNi4cZGfb920iWFPP01OTg4bP/+cB/JHzdtT0jVBzuPfSXAPIp5asclRd/y8mTOZMXGiqsUatjrJSlUeayv+119/ZefOnQWt8x9//PG2bY1GI61bty4I5m3atCE01P5TKAvWu35vsf0uQhtZWVk88sgj/Pvf/wZg9OjRvPPOOwUDOwuzzZH2xap5epSL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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="2.3:-Transformations">2.3: Transformations<a class="anchor-link" href="#2.3:-Transformations">&#182;</a></h3><p>Transformations are a quick way to include standard preprocessing when loading the graphs (e.g., automatically computing edge from the nodes positions). They work pretty much like torchvision's transforms.</p>
<p>See the full list of available transformations here:</p>
<p><a href="https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html">https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html</a></p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># As an experiment, we load the graph with a sparse adjacency format instead of the COO list</span>
<span class="n">mutag_adj</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">TUDataset</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s1">&#39;.&#39;</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">&#39;MUTAG&#39;</span><span class="p">,</span> <span class="n">transform</span><span class="o">=</span><span class="n">ptgeom</span><span class="o">.</span><span class="n">transforms</span><span class="o">.</span><span class="n">ToSparseTensor</span><span class="p">())</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># The format has a number of useful methods that are already implemented: https://github.com/rusty1s/pytorch_sparse</span>
<span class="c1"># For example, we can perform a single step of diffusion on the node features efficiently with a sparse-dense matrix multiplication</span>

<span class="n">mutag_adj</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Data(x=[17, 7], edge_attr=[38, 4], y=[1], adj_t=[17, 17, nnz=38])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">mutag_adj</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">adj_t</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>SparseTensor(row=tensor([ 0,  0,  1,  1,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  6,  6,  7,  7,
                            8,  8,  8,  9,  9,  9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14,
                           15, 16]),
             col=tensor([ 1,  5,  0,  2,  1,  3,  2,  4,  9,  3,  5,  6,  0,  4,  4,  7,  6,  8,
                            7,  9, 13,  3,  8, 10,  9, 11, 10, 12, 11, 13, 14,  8, 12, 12, 15, 16,
                           14, 14]),
             size=(17, 17), nnz=38, density=13.15%)</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>🔥 <strong>Warmup Exercise no. 1</strong></p>
<p>Imagine you are a (probably crazy) chemist and you want to <em>add self loops</em> to all of the molecules in your dataset.</p>
<p>What would you do? Plot a graph after adding self loops. Hint: <a href="https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddSelfLoops.html#torch_geometric.transforms.AddSelfLoops">this transform</a></p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># We load the graph and add self loops to all nodes (probably doesn&#39;t make sense from a chemical point of view 🤔)</span>
<span class="n">mutag_self</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">TUDataset</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s1">&#39;.&#39;</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">&#39;MUTAG&#39;</span><span class="p">,</span> <span class="n">transform</span><span class="o">=</span><span class="n">ptgeom</span><span class="o">.</span><span class="n">transforms</span><span class="o">.</span><span class="n">AddSelfLoops</span><span class="p">())</span>
<span class="n">mutag_self_0</span> <span class="o">=</span> <span class="n">mutag_self</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">draw_molecule</span><span class="p">(</span><span class="n">mutag_self_0</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="2.4-Data-loading">2.4 Data loading<a class="anchor-link" href="#2.4-Data-loading">&#182;</a></h3><p>Data loaders are a nice utility to automatically build mini-batches (either a subset of graphs, or a subgraph extracted from a single graph) from the dataset.</p>
<p>Pytorch Geometric manages the batching by <a href="https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html">stacking the adjacency matrices</a> into a single huge graph.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Plain MUTAG without self loops</span>
<span class="n">mutag</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">TUDataset</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s1">&#39;.&#39;</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">&#39;MUTAG&#39;</span><span class="p">)</span> <span class="c1"># just like a plain Torch Dataset</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># First, we split the original dataset into a training and test spart with a stratified split on the class</span>
<span class="n">train_idx</span><span class="p">,</span> <span class="n">test_idx</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">mutag</span><span class="p">)),</span> <span class="n">stratify</span><span class="o">=</span><span class="p">[</span><span class="n">m</span><span class="o">.</span><span class="n">y</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">item</span><span class="p">()</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">mutag</span><span class="p">],</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.25</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">11</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Build the two loaders</span>
<span class="n">train_loader</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">loader</span><span class="o">.</span><span class="n">DataLoader</span><span class="p">(</span><span class="n">mutag</span><span class="p">[</span><span class="n">train_idx</span><span class="p">],</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">test_loader</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">loader</span><span class="o">.</span><span class="n">DataLoader</span><span class="p">(</span><span class="n">mutag</span><span class="p">[</span><span class="n">test_idx</span><span class="p">],</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Let us inspect the first batch of data</span>
<span class="n">batch</span> <span class="o">=</span> <span class="nb">next</span><span class="p">(</span><span class="nb">iter</span><span class="p">(</span><span class="n">train_loader</span><span class="p">))</span>
<span class="n">batch</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>DataBatch(edge_index=[2, 1258], x=[568, 7], edge_attr=[1258, 4], y=[32], batch=[568], ptr=[33])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># The batch is built by considering all the subgraphs as a single giant graph with unconnected components</span>
<span class="nb">print</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span> <span class="c1"># All the nodes of the 32 graphs are put together</span>
<span class="nb">print</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">y</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span> <span class="c1"># A single label for each graph</span>
<span class="nb">print</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">edge_index</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>torch.Size([568, 7])
torch.Size([32])
torch.Size([2, 1258])
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>🔥 <strong>Warmup Exercise no. 2</strong></p>
<p>As we said, PyTorch Geometric creates batches by stacking together small graphs into a single large one.</p>
<p>Create a dataloader with batch_size = 2 and plot the first batch to check it's content.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Don&#39;t do this with large batch size</span>
<span class="n">draw_molecule</span><span class="p">(</span><span class="nb">next</span><span class="p">(</span><span class="nb">iter</span><span class="p">(</span><span class="n">ptgeom</span><span class="o">.</span><span class="n">loader</span><span class="o">.</span><span class="n">DataLoader</span><span class="p">(</span><span class="n">mutag</span><span class="p">[</span><span class="n">train_idx</span><span class="p">],</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">))))</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># If we built this new huge graph, how do we keep track of all the small subgraphs 🤔 ?</span>
<span class="c1"># There is an additional property linking each node to its corresponding graph index</span>
<span class="nb">print</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">30</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>torch.Size([568])
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 2])
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><img src="https://raw.githubusercontent.com/alessiodevoto/gnns_xai_liverpool/main/images/batching.png" alt=""></p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># We can perform a graph-level average with torch_scatter, see the figure here for a visual explanation:</span>
<span class="c1"># https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html</span>
<span class="n">torch_scatter</span><span class="o">.</span><span class="n">scatter_sum</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([32, 7])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Alternatively, PyG has this implemented as a functional layer</span>
<span class="n">ptgeom</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">global_mean_pool</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">)</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([32, 7])</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="3.-&#129668;-Design-and-train-the-Graph-Neural-Network">3. &#129668; Design and train the Graph Neural Network<a class="anchor-link" href="#3.-&#129668;-Design-and-train-the-Graph-Neural-Network">&#182;</a></h2><p>We have explored the data and created the Dataloaders, which will help us during the training. We are finally able to build the model!</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Layers in PyG are very similar to PyTorch, e.g., this is a standard graph convolutional layer</span>
<span class="n">gc</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">GCNConv</span><span class="p">(</span><span class="n">mutag</span><span class="o">.</span><span class="n">num_features</span><span class="p">,</span> <span class="mi">14</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Pay attention to the forward arguments</span>
<span class="n">gc</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_index</span><span class="p">)</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([568, 14])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Different layers have different properties, see this &quot;cheatsheet&quot; from the documentation:</span>
<span class="c1"># https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html</span>
<span class="c1"># For example, GCNConv accepts an additional &quot;edge_weight&quot; parameter to weight each edge.</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># If you are not used to PyTorch Lightning, see the 5-minutes intro from here:</span>
<span class="c1"># https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html</span>

<span class="n">train_losses</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">eval_accs</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">class</span> <span class="nc">MUTAGClassifier</span><span class="p">(</span><span class="n">ptlight</span><span class="o">.</span><span class="n">LightningModule</span><span class="p">):</span>

  <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">hidden_features</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
    <span class="nb">super</span><span class="p">()</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">gc1</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">GCNConv</span><span class="p">(</span><span class="n">mutag</span><span class="o">.</span><span class="n">num_features</span><span class="p">,</span> <span class="n">hidden_features</span><span class="p">)</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">gc2</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">GCNConv</span><span class="p">(</span><span class="n">hidden_features</span><span class="p">,</span> <span class="n">hidden_features</span><span class="p">)</span>      <span class="c1"># two &quot;hops&quot; seems enough for these small graphs</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">head</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">hidden_features</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>                     <span class="c1"># binary classification</span>

  <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">batch</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">edge_weight</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>

    <span class="c1"># unwrap the graph if the whole Data object was passed</span>
    <span class="k">if</span> <span class="n">edge_index</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
      <span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="p">,</span> <span class="n">batch</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">x</span><span class="o">.</span><span class="n">edge_index</span><span class="p">,</span> <span class="n">x</span><span class="o">.</span><span class="n">batch</span>

    <span class="c1"># GNN layers</span>
    <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">gc1</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="p">,</span> <span class="n">edge_weight</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">gc2</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="p">,</span> <span class="n">edge_weight</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

    <span class="n">x</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">global_mean_pool</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="p">)</span> <span class="c1"># now it&#39;s (batch_size, embedding_dim)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">p</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
    <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">logits</span>

  <span class="k">def</span> <span class="nf">configure_optimizers</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
      <span class="n">optimizer</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">1e-4</span><span class="p">)</span>
      <span class="k">return</span> <span class="n">optimizer</span>


  <span class="k">def</span> <span class="nf">training_step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch</span><span class="p">,</span> <span class="n">_</span><span class="p">):</span>

      <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_index</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">)</span>
      <span class="n">target</span> <span class="o">=</span> <span class="n">batch</span><span class="o">.</span><span class="n">y</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
      <span class="n">loss</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">binary_cross_entropy_with_logits</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="o">.</span><span class="n">float</span><span class="p">())</span>

      <span class="bp">self</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="s2">&quot;train_loss&quot;</span><span class="p">,</span> <span class="n">loss</span><span class="p">)</span>
      <span class="bp">self</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="s2">&quot;train_accuracy&quot;</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">task</span><span class="o">=</span><span class="s1">&#39;binary&#39;</span><span class="p">),</span> <span class="n">prog_bar</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>
      <span class="n">train_losses</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">loss</span><span class="o">.</span><span class="n">detach</span><span class="p">())</span>
      <span class="k">return</span> <span class="n">loss</span>

  <span class="k">def</span> <span class="nf">validation_step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch</span><span class="p">,</span> <span class="n">_</span><span class="p">):</span>

    <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_index</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">)</span>
    <span class="n">target</span> <span class="o">=</span> <span class="n">batch</span><span class="o">.</span><span class="n">y</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">binary_cross_entropy_with_logits</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="o">.</span><span class="n">float</span><span class="p">())</span>

    <span class="bp">self</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="s2">&quot;eval_accuracy&quot;</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">task</span><span class="o">=</span><span class="s1">&#39;binary&#39;</span><span class="p">),</span> <span class="n">prog_bar</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>
    <span class="n">eval_accs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">accuracy</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">task</span><span class="o">=</span><span class="s1">&#39;binary&#39;</span><span class="p">))</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># print the model</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">MUTAGClassifier</span><span class="p">(</span><span class="n">hidden_features</span><span class="o">=</span><span class="mi">256</span><span class="p">)</span>
<span class="n">model</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>MUTAGClassifier(
  (gc1): GCNConv(7, 256)
  (gc2): GCNConv(256, 256)
  (head): Linear(in_features=256, out_features=1, bias=True)
)</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># forward one batch</span>

<span class="n">batch_out</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">batch</span><span class="p">)</span>
<span class="n">batch_out</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([32, 1])</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="3.1-Training-the-model">3.1 Training the model<a class="anchor-link" href="#3.1-Training-the-model">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># We save checkpoints every 50 epochs</span>
<span class="c1"># This is like taking &#39;snapshots&#39; of the model every 50 epochs</span>
<span class="c1"># We will use this in the next notebook</span>

<span class="n">checkpoint_callback</span> <span class="o">=</span> <span class="n">ptlight</span><span class="o">.</span><span class="n">callbacks</span><span class="o">.</span><span class="n">ModelCheckpoint</span><span class="p">(</span>
    <span class="n">dirpath</span><span class="o">=</span><span class="s1">&#39;./checkpoints/&#39;</span><span class="p">,</span>
    <span class="n">filename</span><span class="o">=</span><span class="s1">&#39;gnn-</span><span class="si">{epoch:02d}</span><span class="s1">&#39;</span><span class="p">,</span>
    <span class="n">every_n_epochs</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
    <span class="n">save_top_k</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># define the trainer</span>
<span class="n">trainer</span> <span class="o">=</span> <span class="n">ptlight</span><span class="o">.</span><span class="n">Trainer</span><span class="p">(</span><span class="n">max_epochs</span><span class="o">=</span><span class="mi">80</span><span class="p">,</span> <span class="n">callbacks</span><span class="o">=</span><span class="p">[</span><span class="n">checkpoint_callback</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>INFO:pytorch_lightning.utilities.rank_zero:GPU available: False, used: False
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># This is not a particularly well-designed model, we expect approximately 80% test accuracy</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">train_loader</span><span class="p">,</span> <span class="n">test_loader</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>INFO:pytorch_lightning.callbacks.model_summary:
  | Name | Type    | Params
---------------------------------
0 | gc1  | GCNConv | 2.0 K 
1 | gc2  | GCNConv | 65.8 K
2 | head | Linear  | 257   
---------------------------------
68.1 K    Trainable params
0         Non-trainable params
68.1 K    Total params
0.272     Total estimated model params size (MB)
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="001a5aa8-1ef2-4a2d-999b-3408d98dbd72" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('001a5aa8-1ef2-4a2d-999b-3408d98dbd72');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "3ead607763e34297a116d59a735461a3"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d768aeee-77cc-4df0-af45-b9caedd62351" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d768aeee-77cc-4df0-af45-b9caedd62351');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "817877f898ac45628cc10633f499b7b2"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="03bc827d-bca8-41f9-a99c-14bae4683b8b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('03bc827d-bca8-41f9-a99c-14bae4683b8b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e859e6f83cf34985b31e846f76310cfa"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5662531e-7d67-40c8-a99d-62025bc29697" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5662531e-7d67-40c8-a99d-62025bc29697');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b60f36ac272940b8b4a12938f473eef7"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ec068ef7-ff4f-4fd9-8c67-e4b5c5969c02" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ec068ef7-ff4f-4fd9-8c67-e4b5c5969c02');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "9826a6df263a4cc09146426c5085c1ef"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2b6bd890-39c4-4e81-97ee-9fdcebccb8c8" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2b6bd890-39c4-4e81-97ee-9fdcebccb8c8');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "cd526b935fc14cb28c49610b52867152"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="c3cdce18-1c63-4795-a070-4dec9e07a7ce" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('c3cdce18-1c63-4795-a070-4dec9e07a7ce');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ca491da8f36a425e94883583181e5821"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="01690166-1323-4f98-b4b0-1d5967c5cbd8" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('01690166-1323-4f98-b4b0-1d5967c5cbd8');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "79888acce2b84d6885b8279343c6e99d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8c348549-220b-4226-bdf2-783207d2fe90" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8c348549-220b-4226-bdf2-783207d2fe90');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "127d2d51b19b459bb8a9d41eac314b7d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="82c1fe2b-a0dc-4bed-bf4b-2a336e6fa7b6" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('82c1fe2b-a0dc-4bed-bf4b-2a336e6fa7b6');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e6426c38fbb54ec6b22107b9a042b9a9"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="1cf2c1e0-f609-440b-b830-cdd2e8de57d2" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('1cf2c1e0-f609-440b-b830-cdd2e8de57d2');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "428bfccfc3904f349c7d2b751bdf8d76"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d6e286d9-b1bf-45b6-9272-c9416716b901" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d6e286d9-b1bf-45b6-9272-c9416716b901');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "67f678ce112b4b50886113ff8323cc36"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3065bfae-8705-4a13-951c-5f35c6a5962f" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3065bfae-8705-4a13-951c-5f35c6a5962f');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "8dc28498ca9642dcb78bfac5afe5f099"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5c738f4a-8235-4e83-8b47-2eeeda40b82a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5c738f4a-8235-4e83-8b47-2eeeda40b82a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "68d3ee9a1885477f9d9d112b6ca7f648"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="cdc89f0f-27b3-45fd-961d-f87aae066ee4" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('cdc89f0f-27b3-45fd-961d-f87aae066ee4');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "23bf5d410f124cbd9cac674c5e2d90c1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3cccd3a9-07b3-4c55-b84a-4b34b1c8a895" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3cccd3a9-07b3-4c55-b84a-4b34b1c8a895');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d99434a246c443a1aef1507ed92708e6"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8d924cbf-85d0-43a3-b05a-9f4813debce6" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8d924cbf-85d0-43a3-b05a-9f4813debce6');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d0b3d99ff02644108e26aeb73a4fbfc1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3a83dfc4-e7b5-4ee3-bd76-88ba3c300738" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3a83dfc4-e7b5-4ee3-bd76-88ba3c300738');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "340290e5261842b2b535936882ebefb1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="f10e0e21-4aee-4140-b2a3-10cdfb5b9c7b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('f10e0e21-4aee-4140-b2a3-10cdfb5b9c7b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c3825763f8064428ba9900312c540f9f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="66166adf-139a-4a60-813b-39b523f11e40" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('66166adf-139a-4a60-813b-39b523f11e40');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "f71aa31a6feb4b28bd6b1703032a052b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="95d72399-a7a8-491f-b1f9-0816425fc97c" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('95d72399-a7a8-491f-b1f9-0816425fc97c');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "1e2e89d5104a458f8f666cfcb2ec4a7d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8bb47f91-1766-46f7-98d7-0d304dae3e85" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8bb47f91-1766-46f7-98d7-0d304dae3e85');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "326e1e50d4214f3594a737572b675f01"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8ed9a473-2978-4432-8b0a-0fd8e945a132" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8ed9a473-2978-4432-8b0a-0fd8e945a132');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5053c5f842db4284888418db9d8296f0"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="a7026705-c514-455f-87f2-8e9ae78ce813" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('a7026705-c514-455f-87f2-8e9ae78ce813');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b2531bb217d341de9a573d704802a08e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="c32729d7-ec51-4980-973f-4eb03792a557" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('c32729d7-ec51-4980-973f-4eb03792a557');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2f975c4ae6204081b2974f87890e924a"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="06b43a9d-2eb2-4e66-9b23-bc2b91cafc1b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('06b43a9d-2eb2-4e66-9b23-bc2b91cafc1b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "62551809c1c84b4994400817e95d8e5c"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d8a2013f-2044-4a25-9b88-c6307f8e19ad" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d8a2013f-2044-4a25-9b88-c6307f8e19ad');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "7ae266742b964f5b83beea806162f142"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="316fecfb-8d9c-4235-bf9a-b71e36c2693f" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('316fecfb-8d9c-4235-bf9a-b71e36c2693f');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6cf28da316b8435294b2b393da8f9420"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="07f949c4-0406-4683-97ff-d976e7494be8" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('07f949c4-0406-4683-97ff-d976e7494be8');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "fd487eeac9184db2acc4845054708bda"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="63dda0cf-3578-42ac-a0b6-59f9b862ce6d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('63dda0cf-3578-42ac-a0b6-59f9b862ce6d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "208325d3a2934aeb9fdf602b65c45b6f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b9227f04-bb3e-4b09-941e-4cca5be08263" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b9227f04-bb3e-4b09-941e-4cca5be08263');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "306a5c4a345544c582c0007327968b75"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0b7aed80-c196-40c7-90fa-724f84856a17" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0b7aed80-c196-40c7-90fa-724f84856a17');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5934008662534798954d509674c60923"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d54dad90-2c1c-4577-9a77-452cd12720a8" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d54dad90-2c1c-4577-9a77-452cd12720a8');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2356361b053f409c9002777326bd226f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ca2b7645-0049-40f6-a902-6ce04e8dc0cc" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ca2b7645-0049-40f6-a902-6ce04e8dc0cc');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "7496aaa12b8f4109af7b28658a420e4e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="628d7cb7-ad39-44d7-ae45-afa7d18ea5cf" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('628d7cb7-ad39-44d7-ae45-afa7d18ea5cf');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "395d7e25452641baa0ceb3210163b4cb"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b9caac0b-afce-4dc0-9418-bbe45d80d04e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b9caac0b-afce-4dc0-9418-bbe45d80d04e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d6289892932a483fa47f4eb80ee02315"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ecccd03d-7765-4b21-9841-a7b2ef4cd8ee" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ecccd03d-7765-4b21-9841-a7b2ef4cd8ee');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "84aa19cb7b6449e2a2d055db5ebfd35d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="16c8ddd5-89a2-463f-93d8-c95c711ca1bd" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('16c8ddd5-89a2-463f-93d8-c95c711ca1bd');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c18551483b564db1a999b092da39edfc"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d377de0c-d82d-4e61-aad8-8b8a49e4e550" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d377de0c-d82d-4e61-aad8-8b8a49e4e550');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "0bbc6a48c6114f1d99ea7698ced77487"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="dfaf155b-cf8a-4957-bd5d-5cd3ad1b9220" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('dfaf155b-cf8a-4957-bd5d-5cd3ad1b9220');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b92aa2fed3a143fba09c6be94c1ab4a2"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="a95d18a5-9a98-4980-9f83-47b1e1d76711" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('a95d18a5-9a98-4980-9f83-47b1e1d76711');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "99bc80215d7d41179a6b17f371694dc1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="492d6295-1c2f-40c3-b530-890b0727c41d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('492d6295-1c2f-40c3-b530-890b0727c41d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "f685bc21e1a447e28f22f4184159e539"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0dbe4f7f-4236-4c84-bde6-fc5154be79b0" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0dbe4f7f-4236-4c84-bde6-fc5154be79b0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "f90e9ebcafe54e8a86bf5e800fd4787d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="f21db2e1-0b61-4f28-aa92-95ec501c9ca4" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('f21db2e1-0b61-4f28-aa92-95ec501c9ca4');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "09a686ba43084047a0bea3707d163a1b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0c2828f5-444a-4c87-b46a-f2408e281c8c" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0c2828f5-444a-4c87-b46a-f2408e281c8c');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e5b9c8410ad7470ca2944e6d9332fa1d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="62f78bfe-40f3-4d28-a759-dd3fa71c1f27" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('62f78bfe-40f3-4d28-a759-dd3fa71c1f27');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ade2c621c92748f08a236dd205e4562f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="6b86fa93-08c7-4a4f-a10a-f3dee172f3df" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('6b86fa93-08c7-4a4f-a10a-f3dee172f3df');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "99abd644076341fab5dad46994dc22fd"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="c5737931-64b9-41ee-896e-1a7ab4b522ad" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('c5737931-64b9-41ee-896e-1a7ab4b522ad');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "f1f6f27fcd2b49db938ed9ea4ddcc64a"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="16e20c51-c837-49ca-9fb8-5e3ed68323d0" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('16e20c51-c837-49ca-9fb8-5e3ed68323d0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "7f709868d9ae4e2597b4c4c9298bcb64"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="87f21ff3-c7c1-474a-a4ea-b2bcc93409fd" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('87f21ff3-c7c1-474a-a4ea-b2bcc93409fd');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b867b9810a1644f688616493923d5950"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0fe6f2d5-d631-4546-8807-0187a4f277f9" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0fe6f2d5-d631-4546-8807-0187a4f277f9');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "06f8cc1b945d4eeeb41a2b866bcef0bb"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="33e0ae4c-4000-4be5-b0c6-4439f09e6913" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('33e0ae4c-4000-4be5-b0c6-4439f09e6913');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "676b228952b4419f83ceb7e40660e76b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="9df72f83-d9c1-495b-a64e-bb4d4ece2392" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('9df72f83-d9c1-495b-a64e-bb4d4ece2392');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6659af28884b439985ceec27082d841d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b72969dd-fd7c-40ee-bc3c-a2f1557a0561" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b72969dd-fd7c-40ee-bc3c-a2f1557a0561');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b4a855a7afc74e3680d3d17daee31c40"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="aa087de8-a838-445b-b976-5211ef0a2a3e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('aa087de8-a838-445b-b976-5211ef0a2a3e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ea4315c2ed304e3fae5b1bcc9dc16646"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ea659d6a-608f-4f40-9cff-026d6390d9cd" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ea659d6a-608f-4f40-9cff-026d6390d9cd');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "9e4b138a25d44e60aa0e72ffe09a54b0"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="4ffe08fd-e382-4e45-aeaf-9aff6b1d45ea" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('4ffe08fd-e382-4e45-aeaf-9aff6b1d45ea');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "640814e83d82493f8dac746a8f7e9a91"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="17cd6e83-2e6e-4b1b-b064-2e7c186dddd9" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('17cd6e83-2e6e-4b1b-b064-2e7c186dddd9');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "3e5d1e74763346d7975214db9ed6e892"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="430630b6-f00d-4129-9880-1b0da6d1aa67" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('430630b6-f00d-4129-9880-1b0da6d1aa67');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6657f58e2a284fa7809455a88e5e6fdf"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5b18daaf-1af2-4b47-9f19-6a36cdbc5e21" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5b18daaf-1af2-4b47-9f19-6a36cdbc5e21');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "fffbdd91348545baa95a2bf7cd72a618"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2895511c-ed91-4516-b2bd-c2551fe41be1" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2895511c-ed91-4516-b2bd-c2551fe41be1');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "1c17d238dbf14f4da2d81d873cbd2fa8"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="7d4817a5-8312-46d7-ae60-85c593a589f3" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('7d4817a5-8312-46d7-ae60-85c593a589f3');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "64f67fc241cf4118b1a1cf9db67bef54"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="f46384ff-8dee-41a8-a203-1eec8dbf50c4" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('f46384ff-8dee-41a8-a203-1eec8dbf50c4');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "06597913f6db47faa3e44cb3249509fa"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="01442470-9eac-4839-b598-cf9878eac039" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('01442470-9eac-4839-b598-cf9878eac039');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "11898e167cc64ffcb58692d453f07656"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ccec0859-f6e4-4a5d-babe-4b3736c150c5" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ccec0859-f6e4-4a5d-babe-4b3736c150c5');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "f3582d6678c84c408e7da9bd9e86935e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="cf2d0f68-4621-4a6d-9403-63f894628eea" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('cf2d0f68-4621-4a6d-9403-63f894628eea');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e91d60a15d3a4acfa3b5e59d41c34774"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="cdcd1d7e-aedc-4283-bb17-e44c020611e4" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('cdcd1d7e-aedc-4283-bb17-e44c020611e4');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "503997b64d8a4440b5543080e65a54ca"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="978034d6-f857-48a2-afaa-ead514a8502a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('978034d6-f857-48a2-afaa-ead514a8502a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "0e5c90cd89e743b9a65dd097185faf0f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e5bd420a-8bac-4e3a-91c7-bd1dd0bd6646" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e5bd420a-8bac-4e3a-91c7-bd1dd0bd6646');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "da719a8ce9b048b98f9f4c3f41e2625a"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="438c49eb-1bd6-4fde-a845-a48688256707" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('438c49eb-1bd6-4fde-a845-a48688256707');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "7a31c7691b56488fa46a3bf0ac99e778"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="9c05e8bd-503d-4f7f-a1e1-1b604a3250c3" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('9c05e8bd-503d-4f7f-a1e1-1b604a3250c3');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d6e4f489ca1b4002b3fe4aa0ba5fa3c9"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5209da2d-8b22-49cb-86f9-06ebbb11669b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5209da2d-8b22-49cb-86f9-06ebbb11669b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "1a3425e2614f4905a840ed09717251a7"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0a94cbce-2f52-442a-bd2d-739e3bdc9f39" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0a94cbce-2f52-442a-bd2d-739e3bdc9f39');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2eb24e9bcc404bc09caffa83e76b1f8c"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="141facef-013b-497f-823c-320afe191763" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('141facef-013b-497f-823c-320afe191763');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "32f738036dda484b977532a282e331ff"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="49709c1f-e271-4e2f-b2bf-17e7825d5cf0" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('49709c1f-e271-4e2f-b2bf-17e7825d5cf0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c83254728e3e46f7840edb65483a7e39"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="a2431970-92c0-4e61-82ea-d4be64f29345" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('a2431970-92c0-4e61-82ea-d4be64f29345');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "4e05677f16384df094455c811dd7432d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="dab08222-a119-4c6f-9ee4-b6b244d512f5" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('dab08222-a119-4c6f-9ee4-b6b244d512f5');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5da9c13e627d406d82019f4da5e51169"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="c4771aef-0368-4d0f-a0b5-1967317d7bc1" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('c4771aef-0368-4d0f-a0b5-1967317d7bc1');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ceca7315422f48d086deffa9337e7ba4"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="05e4d580-4856-4753-a868-f22bebd24596" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('05e4d580-4856-4753-a868-f22bebd24596');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c6e0927c421945c191ee7294cfa0c936"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="99f9fdd0-92b5-422c-a56c-d88d9424c31e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('99f9fdd0-92b5-422c-a56c-d88d9424c31e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d49bb4b5da8b4c42bffb65dd6a5fac21"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2aab3411-708d-4e1b-a28b-c25b9772c872" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2aab3411-708d-4e1b-a28b-c25b9772c872');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d082f47015fc42d7bbcb077d4d909310"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="c84dd695-a4b6-4f4d-960e-8dc40d2da08e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('c84dd695-a4b6-4f4d-960e-8dc40d2da08e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6eb19163e78d4451815e28976e2cdad3"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=80` reached.
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># simple plots to visualize metrics</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span><span class="mi">4</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">train_losses</span><span class="p">[::</span><span class="nb">len</span><span class="p">(</span><span class="n">train_losses</span><span class="p">)</span><span class="o">//</span><span class="nb">len</span><span class="p">(</span><span class="n">eval_accs</span><span class="p">)])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">eval_accs</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">([</span><span class="s1">&#39;Loss&#39;</span><span class="p">,</span> <span class="s1">&#39;Accuracy&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApEAAAHxCAYAAAA1N+cMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAABJ0AAASdAHeZh94AAEAAElEQVR4nOx9eZwcVb39qaW32ZNMJvtGFhISQlgDGCKKILvIqqgR5T3eE0FFEUXlPVT8oUZRhAeCImsQNCwiuBD2nbCEJWEJScg+ySSZfaaX6qr6/VF1q+69dau6umfJhNzz+eSTnurqqlvV1XVPnfNdFNu2bUhISEhISEhISEiUAXV3D0BCQkJCQkJCQmLPgySREhISEhISEhISZUOSSAkJCQkJCQkJibIhSaSEhISEhISEhETZkCRSQkJCQkJCQkKibEgSKSEhISEhISEhUTYkiZSQkJCQkJCQkCgbkkRKSEhISEhISEiUDUkiJSQkJCQkJCQkyoYkkRISEhISEhISEmVD390D6A+0t7fj6aefxoQJE5BKpXb3cCQkJCQkJCQk9ijk83ls2rQJH//4x9HQ0BDrMx8JEvn000/jtNNO293DkJCQkJCQkJDYo/Hggw/iM5/5TKx1PxIkcsKECQCcA582bdpuHo2EhISEhISExJ6FNWvW4LTTTvM4VRx8JEgksbCnTZuG2bNn7+bRSEhISEhISEjsmSgnLFAm1khISEhISEhISJQNSSIlJCQkJCQkJCTKhiSREhISEhISEhISZUOSSAkJCQkJCQkJibLxkUiskZCQkJCQkAAsy8L27duRz+dhWdbuHo7EEICqqkilUhg1ahRUtX+1Q0kiJSQkJCQkPgKwLAsbN25ENpuFpmnQNA2KouzuYUnsRti2jUKhgGw2i3w+j4kTJ/YrkZQkUkJCQkJC4iOA7du3I5vNYvjw4WhqapIEUgKAQyRbWlrQ2tqK7du3Y8yYMf22bRkTKSEhISEh8RFAPp+HpmmSQEowUBQFTU1N0DQN+Xy+X7ctSaSEhISEhMRHAJZlSQtbQghFUaBpWr/HyUoSKSEhISEh8RGBJJASYRiIa0OSSAkJCQkJCQkJibIhSaSEhISEhISEhETZkCRSQkJCQkJCQkKibEgSKSEhISEhITHkcdttt0FRFLz66qu7eygSLiSJlJCQkJCQkJCQKBuSREpISEjsJbBtG+9v60KhKNvhSUhI9B2SREpISEjsJbjjxQ349G+fwcV/fn13D0VCYkCwYsUKnHDCCairq0NNTQ2OOeYYvPTSS8w6hmHgxz/+MaZPn450Oo0RI0ZgwYIFWLZsmbfOtm3b8JWvfAXjx49HKpXCmDFj8JnPfAbr168f5CMa2pBtDyUkJCT2Ery5qR0A8Nbmjt07EAmJAcCqVatw1FFHoa6uDpdddhkSiQRuuukmHH300Xj66acxf/58AMCVV16Jq6++Gv/xH/+Bww47DJ2dnXj11Vfx+uuv49hjjwUAnHHGGVi1ahUuvvhiTJ48GS0tLVi2bBk2btyIyZMn78ajHFqQJFJCQkJiL0HRspn/JSQ+SvjRj34EwzDw3HPPYZ999gEALFq0CPvuuy8uu+wyPP300wCARx55BCeeeCJuvvlm4Xba29vxwgsvYPHixbj00ku95ZdffvnAH8QehopJZD6fx//8z//gzjvvRFtbG+bOnYurrrrKY/FRuOeee/DLX/4S77zzDmpra3HqqafiF7/4BRobGysdjoSEhIRECZgueTQlidzr8OO/r8I7Wzt39zAAAPuNrcP/njK7X7dpmiYeffRRnHbaaR6BBIAxY8bg3HPPxR/+8Ad0dnairq4ODQ0NWLVqFT744ANMnz49sK1MJoNkMomnnnoK559/PoYNG9avY/0ooWISed5552Hp0qX41re+henTp+O2227DiSeeiCeffBILFiwI/dyNN96ICy+8EMcccwyuueYabN68Gddeey1effVVvPzyy0in05UOSUJCQkIiAkW3b27RlIk1exve2dqJlz9s3d3DGDDs2LEDvb292HfffQPvzZo1C5ZlYdOmTZg9ezZ+8pOf4DOf+QxmzJiBOXPm4Pjjj8eXvvQlzJ07FwCQSqXwi1/8At/5zncwatQoHH744Tj55JOxaNEijB49erAPbUijIhK5fPly3HPPPYzUu2jRIsyZMweXXXYZXnjhBeHnCoUCfvCDH2DhwoVYtmyZ18fxyCOPxCmnnII//OEPuPjiiys8FAkJCQmJKEglcu/FfmPrdvcQPOzusSxcuBBr167F3/72Nzz66KP44x//iN/85jf4/e9/j//4j/8AAHzrW9/CKaecggcffBD//ve/ccUVV+Dqq6/GE088gQMPPHC3jn8ooSISuXTpUmiahgsuuMBblk6ncf755+MH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Ac5vPpEOX9dTIislkf7ndKlEsiDnpj+vo6GK0fvv7hH4qGAsTzzxBJqbm3HPPffgnnvuCby/ZMkSHHfccf0xOgDhiqRpiufmVCoFVVUD6x577LFobW3F9773PcycORPV1dXYsmULzjvvPFgVxCUvWrQI3/zmN7F582bk83m89NJLuP7668veTn9BksihBk8NkSRSQkIIRonsRwWJbGtQlEhD/Fq4LlEiKxwXdS/RpRLJwvvO94L7bQX28VDCkiVL0NTUhP/7v/8LvHf//ffjgQcewO9//3tMnToVK1eujNzW1KlT8fLLL8MwDCQSCeE6w4YNA+BketPYsCF+/Ozbb7+N1atX4/bbb8eiRYu85cuWLWPW22effQCg5LgB4HOf+xy+/e1v489//jOy2SwSiQTOOeec2GPqb8gSP0MNnp0tb/ASEkIMWEwkUaUG2c4udQx9VSKpz+kKrUTKkBnvux4M9VmiYmSzWdx///04+eSTceaZZwb+XXTRRejq6sJDDz2EM844A2+++aawFA5JmjnjjDOwc+dOoYJH1pk0aRI0TcMzzzzDvH/DDTfEHremacw2yetrr72WWW/kyJFYuHAh/vSnP2HjRjaplk/0aWxsxAknnIC77roLS5YswfHHH4/GxsbYY+pvSCVyqEHa2RIS0fhIxERydnbkuv0XEymVSA6DqT5LVIyHHnoIXV1dOPXUU4XvH3744V7h8bvvvhtLly7FWWedha9+9as4+OCD0draioceegi///3vccABB2DRokW444478O1vfxvLly/HUUcdhZ6eHjz22GO48MIL8ZnPfAb19fU466yzcN1110FRFEydOhUPP/wwWlpaYo975syZmDp1Ki699FJs2bIFdXV1uO+++wKxkQDwu9/9DgsWLMBBBx2ECy64AFOmTMH69evxyCOP4I033mDWXbRoEc4880wAwE9/+tP4J3IAIEnkUIPMzpaQiEY5VnAl2x2UmEgusSbOulbRcSjKzR6VMZHh2JtiIvdgLFmyBOl0Gscee6zwfVVVcdJJJ2HJkiXI5/N49tln8b//+7944IEHcPvtt6OpqQnHHHOMl/iiaRr+8Y9/4Gc/+xnuvvtu3HfffRgxYgQWLFiA/ff34zWvu+46GIaB3//+90ilUjj77LOxePHikgkwBIlEAn//+9/xjW98A1dffTXS6TQ++9nP4qKL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gmH3c7Uc3jsoIm2wbt06jBs3TicIAGDKlCkAVOuyklAUBf/85z8xdOhQ3XhHRwcGDhyIuro6DB48GNdee22/EQxGrrjiCgwcOBCxWAynnnoq3n333fR9H3zwAZLJJE444QTdMVVVVZg0aRLWrVtX6OUWjEQigWeeeQbTpk3DIYccortv/fr1qKmpwYABAzBixAjceuutSCQSxVlogdi2bRva2tpM1wKgvn+I18K6detQU1ODI4880jRPu7+/smPHDgAwvacAwKOPPoqamhrE43EcddRRePLJJwu9PN+x895ZqdfHH/7wB+zZswfz5s2T3n/bbbehtrYWsVgMkydPxquvvlrgFXqD8XO1XN47mBNpg+3bt6OxsdE0ro198cUXhV5SUVm6dCm2bduGn/3sZ+mxxsZGLFq0CMcddxxSqRRefvllLF68GO+//z5WrlyJcLh/XGpVVVU477zzMGvWLAwdOhQfffQR7r77bnzta1/D6tWrceyxx2L79u0AYHnNrFq1qtDLLhivvPIKvvrqK9Mb/mGHHYZTTz0VRx99NA4cOIBly5bh5z//OdavX4+nn366SKv1n1zXwq5du9Dd3Y1oNIrt27dj+PDhppzASnifueuuuzBw4ECceeaZuvFp06bh/PPPx5gxY/DFF1/gV7/6FebNm4e9e/fi6quvLtJqvcXue2elXh9Lly5FNBrFnDlzdOPBYBAzZszAueeei4MOOggbN27EPffcgzPPPBO///3v8c1vfrNIK3aH8XO1XN47+scnu890dnYiGo2axmOxWPr+SuGTTz7Btddei6lTp+Kyyy5LjxsLJi688EKMGzcOP/zhD7Fs2bKST/q2y7Rp03R23Nlnn405c+Zg4sSJuOWWW/Dyyy+nrwera6Y/Xy9PPvkkIpEIzj//fN34kiVLdD9feumlWLhwIR5++GF873vfw4knnljIZRaMXNeCNicajVbs+8ztt9+O1157DYsXL0Z9fb3uvrfeekv387/+67/i+OOPxw9+8ANcfvnliMfjBVypP9h976zE66O9vR0vvfQSZs2aZbo2Ro8ejVdeeUU3dumll+Koo47CjTfeWFYiUva5Wi7vHbSzbRCPx9Hd3W0a13J0+sMbmR127NiBb37zm6irq0vniWbje9/7HoLBIF577bUCrbA4NDc345xzzsGKFSvQ29ubvh6srpn+er3s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
"
>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># not working due to cookies settings in most cases</span>
<span class="o">%</span><span class="n">reload_ext</span> <span class="n">tensorboard</span>
<span class="o">%</span><span class="n">tensorboard</span> <span class="o">--</span><span class="n">logdir</span><span class="o">=/</span><span class="n">content</span><span class="o">/</span><span class="n">lightning_logs</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="4.-&#128170;-Exercise-time">4. &#128170; Exercise time<a class="anchor-link" href="#4.-&#128170;-Exercise-time">&#182;</a></h2><p>Pytorch geometric contains a wide range of possibilities for Graph Convolutional layers. You can find them <a href="https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html">here</a>.</p>
<ol>
<li><p>Instead of the simple <code>GCNConv</code> we used, build a model making use of different layers, e.g. the GATConv. Train the model and compare the results with the ones we obtained. Are they better or worse?</p>
</li>
<li><p>(If we have time) Can we change the forward function of our model and also use edge weights. Is it beneficial for the training ?</p>
</li>
</ol>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="kn">from</span> <span class="nn">torch_geometric.nn</span> <span class="kn">import</span> <span class="n">GATConv</span>

<span class="c1"># Define the new model</span>

<span class="n">train_losses</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">train_accs</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">eval_accs</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">class</span> <span class="nc">MyCoolGNN</span><span class="p">(</span><span class="n">ptlight</span><span class="o">.</span><span class="n">LightningModule</span><span class="p">):</span>

  <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">hidden_features</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">heads</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
    <span class="nb">super</span><span class="p">()</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">gat1</span> <span class="o">=</span> <span class="n">GATConv</span><span class="p">(</span><span class="n">mutag</span><span class="o">.</span><span class="n">num_features</span><span class="p">,</span> <span class="n">hidden_features</span><span class="p">,</span> <span class="n">heads</span><span class="o">=</span><span class="n">heads</span><span class="p">)</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">gat2</span> <span class="o">=</span> <span class="n">GATConv</span><span class="p">(</span><span class="n">hidden_features</span> <span class="o">*</span> <span class="n">heads</span><span class="p">,</span> <span class="n">hidden_features</span><span class="p">,</span> <span class="n">heads</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">concat</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">head</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">hidden_features</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>

  <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">batch</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">edge_weight</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>

    <span class="c1"># unwrap the graph if the whole Data object was passed</span>
    <span class="k">if</span> <span class="n">edge_index</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
      <span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="p">,</span> <span class="n">batch</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">x</span><span class="o">.</span><span class="n">edge_index</span><span class="p">,</span> <span class="n">x</span><span class="o">.</span><span class="n">batch</span>

    <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">gat1</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="p">,</span> <span class="n">edge_weight</span><span class="p">))</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">gat2</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">edge_index</span><span class="p">,</span> <span class="n">edge_weight</span><span class="p">))</span>

    <span class="n">x</span> <span class="o">=</span> <span class="n">ptgeom</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">global_mean_pool</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">logits</span>

  <span class="k">def</span> <span class="nf">configure_optimizers</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
      <span class="n">optimizer</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">1e-3</span><span class="p">)</span>
      <span class="k">return</span> <span class="n">optimizer</span>


  <span class="k">def</span> <span class="nf">training_step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch</span><span class="p">,</span> <span class="n">_</span><span class="p">):</span>

      <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_index</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_weight</span><span class="p">)</span>
      <span class="n">target</span> <span class="o">=</span> <span class="n">batch</span><span class="o">.</span><span class="n">y</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>

      <span class="n">loss</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">binary_cross_entropy_with_logits</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="o">.</span><span class="n">float</span><span class="p">())</span>
      <span class="n">acc</span> <span class="o">=</span> <span class="n">accuracy</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="s1">&#39;binary&#39;</span><span class="p">)</span>

      <span class="bp">self</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="s1">&#39;train loss&#39;</span><span class="p">,</span> <span class="n">loss</span><span class="p">)</span>
      <span class="bp">self</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="s1">&#39;train acc&#39;</span><span class="p">,</span> <span class="n">acc</span><span class="o">.</span><span class="n">item</span><span class="p">(),</span>  <span class="n">prog_bar</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>
      <span class="n">train_losses</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">loss</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span><span class="o">.</span><span class="n">item</span><span class="p">())</span>
      <span class="n">train_accs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">acc</span><span class="o">.</span><span class="n">item</span><span class="p">())</span>

      <span class="k">return</span> <span class="n">loss</span>

  <span class="k">def</span> <span class="nf">validation_step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch</span><span class="p">,</span> <span class="n">_</span><span class="p">):</span>

      <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">x</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_index</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">batch</span><span class="p">,</span> <span class="n">batch</span><span class="o">.</span><span class="n">edge_weight</span><span class="p">)</span>
      <span class="n">target</span> <span class="o">=</span> <span class="n">batch</span><span class="o">.</span><span class="n">y</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
      <span class="n">acc</span> <span class="o">=</span> <span class="n">accuracy</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="s1">&#39;binary&#39;</span><span class="p">)</span>

      <span class="bp">self</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="s1">&#39;eval acc&#39;</span><span class="p">,</span> <span class="n">acc</span><span class="o">.</span><span class="n">item</span><span class="p">(),</span> <span class="n">prog_bar</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>
      <span class="n">eval_accs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">acc</span><span class="p">)</span>
      <span class="k">return</span> <span class="n">acc</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">MyCoolGNN</span><span class="p">(</span><span class="mi">256</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">model</span><span class="p">(</span><span class="nb">next</span><span class="p">(</span><span class="nb">iter</span><span class="p">(</span><span class="n">train_loader</span><span class="p">)))</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>torch.Size([32, 1])</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="c1"># Train (no callbacks needed this time)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">ptlight</span><span class="o">.</span><span class="n">Trainer</span><span class="p">(</span><span class="n">max_epochs</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>

<span class="n">trainer</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">train_loader</span><span class="p">,</span> <span class="n">test_loader</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>INFO:pytorch_lightning.utilities.rank_zero:GPU available: False, used: False
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
INFO:pytorch_lightning.callbacks.model_summary:
  | Name | Type    | Params
---------------------------------
0 | gat1 | GATConv | 5.1 K 
1 | gat2 | GATConv | 131 K 
2 | head | Linear  | 257   
---------------------------------
137 K     Trainable params
0         Non-trainable params
137 K     Total params
0.549     Total estimated model params size (MB)
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="54ddf2b1-ccdb-4f5d-9137-8b85dd34af3e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('54ddf2b1-ccdb-4f5d-9137-8b85dd34af3e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "aba9ec6d4c7048cc9be1c03cef724057"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>/usr/local/lib/python3.10/dist-packages/pytorch_lightning/loops/fit_loop.py:281: PossibleUserWarning: The number of training batches (5) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
  rank_zero_warn(
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="eb23f5c0-8fbe-41f9-bfb6-0338d3831801" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('eb23f5c0-8fbe-41f9-bfb6-0338d3831801');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "0da6f09d3eca40cba7cd67ee50805986"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="09343737-0a87-4258-9202-5b2a69140afd" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('09343737-0a87-4258-9202-5b2a69140afd');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "10fb3c4780d6464ea639c4df7059a0a8"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="65b08a3e-79e3-4c74-a976-1c61982eab7f" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('65b08a3e-79e3-4c74-a976-1c61982eab7f');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "73a56ec11e6c47b0a7cea2bad5f53cfb"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="99c121f0-b44d-4f74-8811-c7271173b037" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('99c121f0-b44d-4f74-8811-c7271173b037');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "edd369bf297547a09e4ac163f5a608e7"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0fade69b-2efd-465e-88af-3031a3091c0a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0fade69b-2efd-465e-88af-3031a3091c0a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "25b352e6f15c41869c07b570754c5ba7"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ac362673-7b08-49df-9f4e-48b2bc7c47f9" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ac362673-7b08-49df-9f4e-48b2bc7c47f9');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "3b84cae9cf684c2198c6335ab5e78247"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="6ea91da3-b402-4a4f-bbb9-d98ab6229c7d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('6ea91da3-b402-4a4f-bbb9-d98ab6229c7d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "822e8e3ac777420c90e93c7ae6931175"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="891b6b5e-5879-4e4c-b69d-93d02f46ce80" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('891b6b5e-5879-4e4c-b69d-93d02f46ce80');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "dece4e8793ab4793a6cf65c1d456a51e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="279238cb-90fb-42e8-8677-8e0de76383ca" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('279238cb-90fb-42e8-8677-8e0de76383ca');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "23b14aea5ab842f0822eadd8b6039ff9"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e7c398bc-e1dd-410d-a17b-d0161ecd2050" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e7c398bc-e1dd-410d-a17b-d0161ecd2050');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "827859747529414998c8bdd95fbdefd7"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3332652c-e769-426b-a34e-baa1ffd50c77" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3332652c-e769-426b-a34e-baa1ffd50c77');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c865e3c0fe5f4c5fa5892cb373521d78"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="71e57159-5bc9-4730-be1c-b486ce0cd8e2" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('71e57159-5bc9-4730-be1c-b486ce0cd8e2');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "113183fcbc494277bedf99e2cbfeef38"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b92fe5f3-0891-4430-b0d1-94e4942cd1f1" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b92fe5f3-0891-4430-b0d1-94e4942cd1f1');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d353d3eedd864b2ebde566c23f0bc97e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2b10a08d-db6c-48a8-9ac0-b1631653ebbf" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2b10a08d-db6c-48a8-9ac0-b1631653ebbf');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "bb7d4fb67f8b4d1ca2086ce30f98c7be"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="147aeb28-24ec-408a-93ed-7c2257ce6db2" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('147aeb28-24ec-408a-93ed-7c2257ce6db2');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "788885ca8db44083b6ccace49ebe8b69"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="fefa9623-d8d6-44c7-be3c-eaf11101a9ff" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('fefa9623-d8d6-44c7-be3c-eaf11101a9ff');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "98149bf3967542c9b37f4c24c0764cae"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="dcb1b38d-296a-4397-86be-09e034c815d3" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('dcb1b38d-296a-4397-86be-09e034c815d3');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6ae5f0c54c3c4bb498f19f6c637053c5"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d4803374-035f-4920-bcc3-9fc84db635d7" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d4803374-035f-4920-bcc3-9fc84db635d7');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "da3a0d900cd34f04a235383d155fffe2"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b2972aa3-db2d-4fd7-88a1-0c3f34122125" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b2972aa3-db2d-4fd7-88a1-0c3f34122125');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d10f47e2d3644b1e939fabee16f069b0"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e0f2dfe4-ac12-4a8e-9428-9859c595a558" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e0f2dfe4-ac12-4a8e-9428-9859c595a558');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "cb14bfba541d42dc86ab3a6afa08ff73"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3f680174-96de-4763-96a3-d493ceb3045a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3f680174-96de-4763-96a3-d493ceb3045a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5384fed1bf72437593011b6e2c2f0b8c"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d19db23e-afa1-4bdc-b8a0-488eb7973dfe" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d19db23e-afa1-4bdc-b8a0-488eb7973dfe');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "a502664f2ea84ee781fb73d7f3c81e69"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="21c2a5b0-23b8-46d6-b009-923b2c03f4f8" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('21c2a5b0-23b8-46d6-b009-923b2c03f4f8');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "27fda74023f64f4e8e3e1792657a9d7b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="21b5a1b3-0b4b-406b-ab9b-23f0ea8ede5a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('21b5a1b3-0b4b-406b-ab9b-23f0ea8ede5a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "87fac226afe84f6f92c2d1bc3b381172"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b236e3df-9e21-4585-b3cc-206d907ae9e1" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b236e3df-9e21-4585-b3cc-206d907ae9e1');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5d1869689dc3476badb671c631d85c03"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2986fe2d-1c4c-4cd6-9c01-4f3ededa9a99" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2986fe2d-1c4c-4cd6-9c01-4f3ededa9a99');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "01b32c8c357d48dcba30abf0f6c8d89b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="1a3cc50d-c359-4149-9b87-ba3649a20b03" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('1a3cc50d-c359-4149-9b87-ba3649a20b03');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e95778e909534a54a6742419df9586a0"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e494f151-8574-4742-b0be-b407ae92fb56" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e494f151-8574-4742-b0be-b407ae92fb56');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "25460fab50c34e9a8e74b3b919f4fe1d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0a2b6cd4-fc0c-472a-b4d1-6859a1d359a4" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0a2b6cd4-fc0c-472a-b4d1-6859a1d359a4');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "3e4aa97e43464022bab3a6a913429459"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3ed97205-131e-49b6-a147-8024a4cf1649" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3ed97205-131e-49b6-a147-8024a4cf1649');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5c4c2a3183af4c939517d5447183f51d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="660572d2-9031-45d4-bdd6-252839f658a0" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('660572d2-9031-45d4-bdd6-252839f658a0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "a78544972c324ef79b96e019347dd621"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8a512e86-5fdf-4be5-821c-88765a53cd56" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8a512e86-5fdf-4be5-821c-88765a53cd56');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b588721f48de445083ec4386cba54d81"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="04c52d81-51d9-4b12-b8dd-061fccc663bb" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('04c52d81-51d9-4b12-b8dd-061fccc663bb');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2fbb52d53fcb40879516079e12c915a1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b6459292-f351-4647-8765-665d3a0a60fa" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b6459292-f351-4647-8765-665d3a0a60fa');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "4ae0a5133869457c89ce4122031977cc"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="76ea9af0-dbff-4499-850c-8c43c08f6cbe" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('76ea9af0-dbff-4499-850c-8c43c08f6cbe');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "3bebbaf95d784636a8e0177a09164d5d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="752644a8-b749-430e-91ff-7af20f861050" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('752644a8-b749-430e-91ff-7af20f861050');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "34af117f0b194f62b92aa365ba17e0d8"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2764f709-fa73-4d68-a981-f1135b31e2c9" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2764f709-fa73-4d68-a981-f1135b31e2c9');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "76e61843469c4df7af1b5f87a5678817"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="f99bf4bd-120a-4ef4-90d2-ccf1c4a4c66e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('f99bf4bd-120a-4ef4-90d2-ccf1c4a4c66e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "fcc1e17d528c4c75a30ee7e2f73c73ff"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="778d5bcd-1842-494b-aeee-9b1740465f79" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('778d5bcd-1842-494b-aeee-9b1740465f79');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e1fc870716d1483baea75210c312265b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="74c336b7-1277-42f6-ad1f-7658dcf0c888" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('74c336b7-1277-42f6-ad1f-7658dcf0c888');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "1062e9b9cf6d40e0ba3e7afa493d5b58"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="26d536b1-1b79-4e74-8351-c2067704cfdf" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('26d536b1-1b79-4e74-8351-c2067704cfdf');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ccad7e2a130544ea8160c48c4e34606b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e553e3ee-31de-434d-b0f1-501e166aac7c" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e553e3ee-31de-434d-b0f1-501e166aac7c');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6864e063f0914a9d9a23890a907afc9b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b06b5fc1-4295-4007-a526-95e00552f728" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b06b5fc1-4295-4007-a526-95e00552f728');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e831580a0d914b04a15af7bfeac09fd1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="368a4ede-517d-4ccb-8819-a4b7f3feaf91" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('368a4ede-517d-4ccb-8819-a4b7f3feaf91');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "98f4ec29107144869af7cc2987df109f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="387ace2c-cc0c-4b52-bf3a-27692f1edbc0" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('387ace2c-cc0c-4b52-bf3a-27692f1edbc0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d873a842c23e4df59f07dc3dd90ca009"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="4e19f11e-53ba-4012-ace2-a4fa824631be" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('4e19f11e-53ba-4012-ace2-a4fa824631be');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ce9c797ce117404ea55b12fd070b0b64"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="fa78acd7-88b6-42d6-8aec-35f69ea85737" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('fa78acd7-88b6-42d6-8aec-35f69ea85737');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "8358b60a1d8a4f3eadf8a822fc211c17"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0648d6cc-24dd-4fa8-8500-a115b8f7fb78" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0648d6cc-24dd-4fa8-8500-a115b8f7fb78');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "8c433e0254cb4d059b70b73218d642d6"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8da0ee1a-f12d-40d2-a91d-5ced0efaef84" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8da0ee1a-f12d-40d2-a91d-5ced0efaef84');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d85058ef846443618a2f9848fb733861"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2a9ef30e-9313-4202-8a48-32b894dedffe" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2a9ef30e-9313-4202-8a48-32b894dedffe');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d007eaf138a745a4ac329da91cdeda14"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="bd947202-ee0e-4bd3-b9c2-08cea868e17d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('bd947202-ee0e-4bd3-b9c2-08cea868e17d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6fe58cd7d39343a9863a2ef69219fe75"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3baccba3-70aa-4a62-966e-f1255f16e827" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3baccba3-70aa-4a62-966e-f1255f16e827');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c68748cf14cf4c32b6d9bd9818ee3945"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="eb27d1b0-30b2-4f6f-b68c-f13f3682e77c" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('eb27d1b0-30b2-4f6f-b68c-f13f3682e77c');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "1fe5a8349fa54c59a78c8a774fe383d2"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e1396b9d-715c-48e8-ba15-534feffe686c" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e1396b9d-715c-48e8-ba15-534feffe686c');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b560efda079e40cfbc357b50dcd5317f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="1ae885ec-4878-4095-b14f-7cb26ed6f68f" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('1ae885ec-4878-4095-b14f-7cb26ed6f68f');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "25d21d2d258f4f219091048b300ea767"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="50b77a8c-d4ee-4263-b732-465ae7ae3a36" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('50b77a8c-d4ee-4263-b732-465ae7ae3a36');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "fb83513d442b47109fc43d6445468c8e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b869d583-7276-4bc1-b8af-306867699fcf" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b869d583-7276-4bc1-b8af-306867699fcf');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "39593b3d83b542fb89522012910283cb"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="52b1cb67-af29-48f4-8d5b-9ed177775c67" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('52b1cb67-af29-48f4-8d5b-9ed177775c67');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "9c6e15efb604426682a716454afb8d1e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="9c3d06bd-9758-47e0-b1d1-c18ade7e0352" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('9c3d06bd-9758-47e0-b1d1-c18ade7e0352');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "21f3a7413f23484bb08f948aaf3bcbf5"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="537dd7e4-a8c3-4473-a589-3eed3edeee38" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('537dd7e4-a8c3-4473-a589-3eed3edeee38');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5cf6f5fec35742a388f1eeba653c36bc"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="1ef587c6-3eda-41eb-a6c2-0b2f480d5569" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('1ef587c6-3eda-41eb-a6c2-0b2f480d5569');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "0ee7b61821c14e0a9334ca8b75608a55"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="a5163e62-a301-4a0e-9e7b-ec6662e3ff66" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('a5163e62-a301-4a0e-9e7b-ec6662e3ff66');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "bedce267132246f38df73fa6b7877a5c"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="6d727e57-c76f-4fa2-9230-22a9f81d15a3" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('6d727e57-c76f-4fa2-9230-22a9f81d15a3');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c32b8d831ac9467b939e6b8790cc947c"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ff77b9b9-bf62-4a58-98d0-70a3284cf699" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ff77b9b9-bf62-4a58-98d0-70a3284cf699');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ff20c3c354294c83ab318bef941fa98c"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8cfcc7e1-f42e-4f1b-b09d-43b5a34a5e45" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8cfcc7e1-f42e-4f1b-b09d-43b5a34a5e45');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "bd720b8902774188847288a078c56be6"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e71ad559-fe10-467d-9ada-3eebf24faa52" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e71ad559-fe10-467d-9ada-3eebf24faa52');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ee94e47f898448d1b9c74e61822c2a01"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2fd005f0-1c72-47c7-bbcd-2d9371208f2a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2fd005f0-1c72-47c7-bbcd-2d9371208f2a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "02065810ab2441a1a1e9c305c362a7a6"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="7ba94e2a-9ddf-41df-8955-fc381bd70072" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('7ba94e2a-9ddf-41df-8955-fc381bd70072');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "87330e8e79384d45809273f36ce65977"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="adc10b7f-69aa-4384-9fec-47a8bce22e29" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('adc10b7f-69aa-4384-9fec-47a8bce22e29');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "744b173d76634462976e2246a6dae969"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="e3990b4e-872c-4754-9eda-ae0ee7388c00" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('e3990b4e-872c-4754-9eda-ae0ee7388c00');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "427bee8b74b44dab80de87895f9d33fc"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="7fb1a02d-792f-46da-99d7-d53143c234af" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('7fb1a02d-792f-46da-99d7-d53143c234af');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2d7328d35d3444859b4fc67fada6b788"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b193b551-de74-4786-88d3-7623c8f2b0ad" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b193b551-de74-4786-88d3-7623c8f2b0ad');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "e725c0f2dfcc4df1a59b9dafb737661a"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="57ce1e57-04d9-4842-a3bf-91e4fda6ab63" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('57ce1e57-04d9-4842-a3bf-91e4fda6ab63');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "4c8d9487560d440fbebc9c4847e9ab91"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="9e72368d-f721-4a2e-874e-8ed7566ecb4d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('9e72368d-f721-4a2e-874e-8ed7566ecb4d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "93209322c08d4ea793234d6cd898a67d"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d9416f85-4492-4ee4-9ee9-fc71040be53c" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d9416f85-4492-4ee4-9ee9-fc71040be53c');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "a257b937d2d049c2a219528d50c0879f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="568ec7f9-3a6c-4827-9a86-f21133c2073d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('568ec7f9-3a6c-4827-9a86-f21133c2073d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "ad97f735f734459faa6ff649a4f03b6e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="cfe2cb75-e865-418d-ad2b-25ab9fec777a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('cfe2cb75-e865-418d-ad2b-25ab9fec777a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "bb0f671afaae4b48a5df61e621dbee34"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="2094cd13-5bcc-4638-a740-e0fc681b1c2e" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('2094cd13-5bcc-4638-a740-e0fc681b1c2e');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5902371a104b4f16a3f69bddf64cfa3e"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="41e7978f-ea9b-403a-812f-a1717957d549" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('41e7978f-ea9b-403a-812f-a1717957d549');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "11e9b9c80d3948fea66840bc08adc239"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="8b546a49-e66a-4278-80b9-c43a6567b329" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('8b546a49-e66a-4278-80b9-c43a6567b329');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "40341ee0f0ea4d8a98f3cce7e757fad9"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d1d4a9be-9fc7-4409-a199-170cc32eeea0" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d1d4a9be-9fc7-4409-a199-170cc32eeea0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "98367a38f26541c68bd92df9953b9959"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0c2effd4-dbbc-4f51-8418-f34856b2eda6" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0c2effd4-dbbc-4f51-8418-f34856b2eda6');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "45448d76856f4ac4be5e6aeee63fd96b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5c3b0b0e-6d27-4f9f-838f-4d3724148e6d" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5c3b0b0e-6d27-4f9f-838f-4d3724148e6d');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5db7b3f6282f494dae0307f03a836d75"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="f4a174ad-7356-4b05-a179-16ed44c64ca7" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('f4a174ad-7356-4b05-a179-16ed44c64ca7');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d0de86cb57354532bcb2b8a59083dc90"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5e4b832b-dd62-411b-a9df-a9828053f12b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5e4b832b-dd62-411b-a9df-a9828053f12b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "393e4a10f25945fd8f10b758c09502fe"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="66bd28b0-fb3a-4ce8-a144-e2bfb5d2a0b7" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('66bd28b0-fb3a-4ce8-a144-e2bfb5d2a0b7');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c2ca2541a43a41b7b99caa80ef85c7d0"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="4b846259-71e2-4524-a077-b62f6c2e1a46" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('4b846259-71e2-4524-a077-b62f6c2e1a46');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "17ff9a4cbe214006b8d57afb41a64f79"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="d338594f-8139-401f-9db0-1f8af0fa2f9b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('d338594f-8139-401f-9db0-1f8af0fa2f9b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2292b8a22cb64bd08537314563d92dcd"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="4747cff7-3f0f-4ebe-b629-32053083b26f" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('4747cff7-3f0f-4ebe-b629-32053083b26f');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "af58420e539140a0a66dc92a2ed60cbe"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="949e04ce-4d21-4a0c-88a2-6d8e5c9e811b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('949e04ce-4d21-4a0c-88a2-6d8e5c9e811b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "f8fa335dfb474c32ad716e4164b54454"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="3173d58d-ccf1-4a34-bf87-5f508c86ed74" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('3173d58d-ccf1-4a34-bf87-5f508c86ed74');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6969a9bbb2444347af3f07568e0b8395"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="ccaa06f5-088c-44c3-b7d0-95a33df5c5a9" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('ccaa06f5-088c-44c3-b7d0-95a33df5c5a9');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "59101b2849a64afe97ac747777754160"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="da2fdee3-7fbe-414d-ad5f-fd533764cf2b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('da2fdee3-7fbe-414d-ad5f-fd533764cf2b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "c6d3a2634935464c9ede11bdaa96444f"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="62610c01-25ae-4af6-9aaf-386e49631af6" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('62610c01-25ae-4af6-9aaf-386e49631af6');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "acb3317cb0a34dc2b0c9a39852cbde9b"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0bde93df-fb43-41a7-8950-a41ea17b2e90" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0bde93df-fb43-41a7-8950-a41ea17b2e90');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "8b79f6bd4c06484e9dd548aabbf59386"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="b8f92399-d175-4cc3-945b-a66924556f70" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('b8f92399-d175-4cc3-945b-a66924556f70');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "6c58f610052a47f5855674b4eaf9a971"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="a1b217e3-7c0c-4703-9b6e-24f1bf3cc039" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('a1b217e3-7c0c-4703-9b6e-24f1bf3cc039');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "2fd6d910b29f4b8d8519de78c85b2392"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="182634cf-2619-4c4e-89a3-93041ab53a8a" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('182634cf-2619-4c4e-89a3-93041ab53a8a');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "30fa84d68d1745c7b06ade1a90597b95"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="5bd5d64a-bbb2-4853-918b-1a06e30bda7b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('5bd5d64a-bbb2-4853-918b-1a06e30bda7b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "d688dd09372747e4a76cbe9083007a87"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="0828664c-4e35-4da4-94c0-28edb54e4a66" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('0828664c-4e35-4da4-94c0-28edb54e4a66');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "4c3100a2b0eb4fa4975763d8b586f692"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>







<div id="847669c6-81fe-417e-9740-9c9dd12d770b" class="jupyter-widgets jp-OutputArea-output ">
<script type="text/javascript">
var element = document.getElementById('847669c6-81fe-417e-9740-9c9dd12d770b');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "fa64b4b200e349e4992a002df53838a1"}
</script>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr">
<pre>INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=100` reached.
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span><span class="mi">4</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">train_losses</span><span class="p">[::</span><span class="nb">len</span><span class="p">(</span><span class="n">train_losses</span><span class="p">)</span><span class="o">//</span><span class="nb">len</span><span class="p">(</span><span class="n">eval_accs</span><span class="p">)])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">eval_accs</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">train_accs</span><span class="p">[::</span><span class="nb">len</span><span class="p">(</span><span class="n">train_losses</span><span class="p">)</span><span class="o">//</span><span class="nb">len</span><span class="p">(</span><span class="n">eval_accs</span><span class="p">)])</span>

<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">([</span><span class="s1">&#39;Loss&#39;</span><span class="p">,</span> <span class="s1">&#39;Eval Accuracy&#39;</span><span class="p">,</span> <span class="s1">&#39;Train Accuracy&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
"
>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
</body>






<script type="application/vnd.jupyter.widget-state+json">
</script>


</html>