# Frontend Wireframes

## 1) Upload + Parsing Status
```text
+--------------------------------------------------------------------------------+
| Deep Research Enterprise                                   [Project: Alpha v]  |
+--------------------------------------------------------------------------------+
| [ Drag & drop files ] [Browse]                     Parsing Queue: 3 running     |
|------------------------------------------------------------------------------- |
| File Name              Type      Size    Status         Actions                 |
| annual_report.pdf      PDF       8.2MB   Indexed        View | Delete           |
| q4_metrics.xlsx        XLSX      1.1MB   Parsing 72%    View | Cancel           |
| plant_photo.jpg        Image     3.4MB   OCR Pending    View | Delete           |
+--------------------------------------------------------------------------------+
```

## 2) Research Conversation + Evidence Pane
```text
+--------------------------------------------------------------------------------+
| Query: [ How did Q4 revenue change and what are key risks?             ][Ask] |
+--------------------------------------------------------------------------------+
| Conversation (center)                         | Evidence (right pane)           |
|-----------------------------------------------+---------------------------------|
| Assistant: concise answer with caveats        | [1] annual_report.pdf p.14 ¶2   |
| - Revenue grew 22% in Q4 [1]                  | snippet... [View][Copy][Pin]    |
| - Key risk: supply chain volatility [2]       | score: 0.88                     |
|                                               | [2] annual_report.pdf p.15 ¶1   |
| Follow-ups: Compare by region?                | snippet... [View][Copy][Pin]    |
+--------------------------------------------------------------------------------+
```

## 3) File Library + Search
```text
+--------------------------------------------------------------------------------+
| Library Search: [ volatility ] filters: [pdf] [date] [owner]                  |
+--------------------------------------------------------------------------------+
| Results                                                                        |
| - annual_report.pdf > p.15 ¶1: "Risk remains supply chain volatility..."      |
| - board_notes.docx > Heading: Risk Register                                    |
+--------------------------------------------------------------------------------+
```

## 4) Document Viewer with Jump-to-citation
```text
+--------------------------------------------------------------------------------+
| annual_report.pdf                                       page 15 [<] [>]         |
+--------------------------------------------------------------------------------+
| ... text ... [HIGHLIGHTED CHUNK] Risk remains supply chain volatility ...      |
| Citation metadata: file=annual_report.pdf page=15 paragraph=1 offsets=120-238 |
+--------------------------------------------------------------------------------+
```

## 5) Branded Social Bar (Footer)
```text
+--------------------------------------------------------------------------------+
| [📸 Instagram] [💼 LinkedIn] [🐙 GitHub] [🧪 CodePen] [✉️ admin@ladestack.in] |
| [🌐 ladestack.in]                                                               |
+--------------------------------------------------------------------------------+
```
