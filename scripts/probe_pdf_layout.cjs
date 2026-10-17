/* Browser half of the CPU layout probe. No live viewer, API, or database.
 * Usage: node scripts/probe_pdf_layout.cjs OUTPUT_DIR PDFJS_DIR
 * PDFJS_DIR contains pdf.mjs, pdf.worker.mjs, pdf_viewer.css, package.json.
 * PLAYWRIGHT_MODULE and CHROMIUM_PATH select local browser tooling.
 */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const output = path.resolve(process.argv[2]);
  const pdfjs = path.resolve(process.argv[3]);
  const manifest = JSON.parse(fs.readFileSync(path.join(output, 'manifest.json')));
  const pdfjsVersion = JSON.parse(fs.readFileSync(path.join(pdfjs, 'package.json'))).version;
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true, args: ['--no-sandbox']});
  const report = {kind:'hades-pdf-layout-probe-results', version:1,
    docling_version:manifest.docling_version, pdfjs_version:pdfjsVersion, cases:[]};
  try {
    for (const [index, fixture] of manifest.cases.entries()) {
      const pdf = fs.readFileSync(fixture.pdf);
      if (crypto.createHash('sha256').update(pdf).digest('hex') !== fixture.source_sha256)
        throw Error('PDF changed since Docling extraction');
      const doc = JSON.parse(fs.readFileSync(path.join(output, fixture.docling)));
      const page = await browser.newPage({viewport:{width:1100,height:1500}});
      await page.route('**/*', async route => {
        const pathname = new URL(route.request().url()).pathname;
        if (pathname === '/') return route.fulfill({contentType:'text/html', body:
          '<link rel="stylesheet" href="/pdf_viewer.css"><style>body{margin:0}#sheet{position:relative}canvas{display:block}.box{position:absolute;border:1px solid #c00;pointer-events:none}</style><div id="sheet"><canvas></canvas><div class="textLayer"></div></div>'});
        if (pathname === '/source.pdf') return route.fulfill({contentType:'application/pdf',body:pdf});
        if (['/pdf.mjs','/pdf.worker.mjs','/pdf_viewer.css'].includes(pathname))
          return route.fulfill({contentType:pathname.endsWith('css')?'text/css':'text/javascript',
            body:fs.readFileSync(path.join(pdfjs, pathname.slice(1)))});
        return route.abort();
      });
      await page.goto('http://hades-layout.test/');
      const result = await page.evaluate(async ({doc, fixture}) => {
        const pdfjs = await import('/pdf.mjs');
        pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs';
        const pdf = await pdfjs.getDocument({data:new Uint8Array(await (await fetch('/source.pdf')).arrayBuffer()),
          useSystemFonts:true}).promise;
        const page = await pdf.getPage(fixture.page);
        const scale = 1.5, viewport = page.getViewport({scale});
        const sheet = document.querySelector('#sheet'), canvas = sheet.querySelector('canvas');
        sheet.style.width = `${viewport.width}px`;
        sheet.style.height = `${viewport.height}px`;
        sheet.style.setProperty('--scale-factor', scale);
        canvas.width = viewport.width; canvas.height = viewport.height;
        await page.render({canvasContext:canvas.getContext('2d'),viewport}).promise;
        const content = await page.getTextContent();
        const container = sheet.querySelector('.textLayer');
        await new pdfjs.TextLayer({textContentSource:content,container,viewport}).render();
        const normalize = text => text.normalize('NFKC').replace(/\s+/g,' ').trim();
        const allText = content.items.map(item=>item.str || '').join(' ');
        const normalized = normalize(allText), spans = [...container.querySelectorAll('span')];
        const checks = [];
        for (const item of doc.texts) {
          const prov = item.prov?.find(p=>p.page_no===fixture.page);
          if (!prov) continue;
          const box = prov.bbox;
          const height = doc.pages[String(fixture.page)].size.height;
          const top = box.coord_origin === 'BOTTOMLEFT' ? height-box.t : box.t;
          const bottom = box.coord_origin === 'BOTTOMLEFT' ? height-box.b : box.b;
          const rect = {left:box.l*scale,right:box.r*scale,top:top*scale,bottom:bottom*scale};
          const overlay = document.createElement('div'); overlay.className='box';
          Object.assign(overlay.style,{left:`${rect.left}px`,top:`${rect.top}px`,width:`${rect.right-rect.left}px`,height:`${rect.bottom-rect.top}px`});
          sheet.append(overlay);
          const quote = normalize(item.text || '');
          const matches = quote ? normalized.split(quote).length-1 : 0;
          checks.push({id:item.self_ref,label:item.label,docling_text_characters:(item.text||'').length,
            normalized_pdfjs_occurrences:matches,box:rect});
        }
        // Select one prose paragraph/caption from the actual DOM text layer,
        // using the Docling box only to identify candidate spans for this probe.
        const target = checks.find(c=>['text','caption'].includes(c.label) && c.docling_text_characters>60);
        if (!target) throw Error('No prose candidate for a browser-selection check');
        const selected = spans.filter(span=>{
          const r=span.getBoundingClientRect(), x=(r.left+r.right)/2, y=(r.top+r.bottom)/2;
          return x>=target.box.left && x<=target.box.right && y>=target.box.top-2 && y<=target.box.bottom+2;
        });
        if (!selected.length) throw Error('No text-layer spans intersect the selected paragraph');
        const range=document.createRange(); range.setStartBefore(selected[0]); range.setEndAfter(selected.at(-1));
        const selection=window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
        const exact=selection.toString();
        const before=document.createRange();before.selectNodeContents(container);before.setEnd(range.startContainer,range.startOffset);
        const after=document.createRange();after.selectNodeContents(container);after.setStart(range.endContainer,range.endOffset);
        const anchor={source_sha256:fixture.source_sha256,page:fixture.page,exact,
          prefix:before.toString().slice(-100),suffix:after.toString().slice(0,100)};
        if (!exact) throw Error('Browser selection returned no text');
        return {label:fixture.label,page:fixture.page,source_sha256:fixture.source_sha256,
          pdfjs_text_runs:content.items.length,checks,selection:{target:target.id,spans:selected.length,anchor},
          figure_caption_refs:(doc.pictures||[]).map(p=>({id:p.self_ref,captions:p.captions,provenance:p.prov}))};
      }, {doc,fixture});
      await page.screenshot({path:path.join(output,`case-${index}.png`),fullPage:true});
      report.cases.push(result);
      await page.close();
    }
    fs.writeFileSync(path.join(output,'browser-report.json'),JSON.stringify(report,null,2));
    for (const c of report.cases) console.log(c.label,JSON.stringify({elements:c.checks.length,
      unique:c.checks.filter(x=>x.normalized_pdfjs_occurrences===1).length,
      empty:c.checks.filter(x=>!x.docling_text_characters).length,selectedSpans:c.selection.spans}));
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
