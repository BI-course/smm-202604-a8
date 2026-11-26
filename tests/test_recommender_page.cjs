// Execute the page script with a minimal DOM adapter. This checks behavior,
// including a live API request, without claiming full browser/visual coverage.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const test = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../frontend/music_track_recommender.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function setup(fetchResponse = async () => ({ok: true, json: async () => ({recommendations: []})})) {
  function element() {
    return {
      value: '', textContent: '', className: '', disabled: false, children: [],
      focus() { this.focused = true; },
      replaceChildren() { this.children = []; this.textContent = ''; },
      append(...items) { this.children.push(...items); },
      addEventListener(_, handler) { this.handler = handler; },
    };
  }
  const elements = {};
  for (const id of ['predictForm', 'api_base', 'track_ids', 'top_n', 'result', 'submitBtn']) elements[id] = element();
  elements.top_n.value = '3';
  elements.track_ids.value = 'M01, M03';
  const calls = [];
  const context = {
    document: {getElementById: id => elements[id], createElement: element},
    window: {location: {protocol: 'file:'}},
    URL, AbortController, setTimeout, clearTimeout,
    fetch: async (...args) => { calls.push(args); return fetchResponse(...args); },
  };
  vm.runInNewContext(script, context);
  return {elements, calls, submit: () => elements.predictForm.handler({preventDefault() {}})};
}

test('blank and comma-only track lists show a clear error before any request', async () => {
  for (const value of ['', '   ', ', ,']) {
    const page = setup();
    page.elements.track_ids.value = value;
    await page.submit();
    assert.match(page.elements.result.textContent, /at least one track ID/);
    assert.equal(page.calls.length, 0);
    assert.equal(page.elements.track_ids.focused, true);
  }
});

test('invalid or missing counts are rejected', async () => {
  for (const value of ['', '0', '1.5', '21']) {
    const page = setup();
    page.elements.top_n.value = value;
    await page.submit();
    assert.match(page.elements.result.textContent, /whole number/);
    assert.equal(page.calls.length, 0);
  }
});

test('invalid API address is rejected', async () => {
  const page = setup();
  page.elements.api_base.value = 'file:///api';
  await page.submit();
  assert.match(page.elements.result.textContent, /valid API address/);
  assert.equal(page.calls.length, 0);
});

test('deduplicates track IDs and renders track names as text', async () => {
  const page = setup(async () => ({ok: true, json: async () => ({
    model: 'Music Apriori', recommendations: [{track_id: 'M02', track_name: '<b>Track</b>', confidence: .8, lift: 2, support: .1}],
  })}));
  page.elements.track_ids.value = ' M01, M03, M01 ';
  await page.submit();
  assert.deepEqual(JSON.parse(page.calls[0][1].body), {track_ids: ['M01', 'M03'], top_n: 3});
  const item = page.elements.result.children[0].children[0];
  assert.equal(item.children[0].textContent, '<b>Track</b> (M02)');
  assert.equal(page.elements.submitBtn.disabled, false);
});

test('server validation errors and empty results are readable', async () => {
  const failure = setup(async () => ({ok: false, json: async () => ({error: 'Unknown track IDs', unknown_track_ids: ['BAD']})}));
  await failure.submit();
  assert.equal(failure.elements.result.textContent, 'Unknown track IDs: BAD');
  const empty = setup();
  await empty.submit();
  assert.match(empty.elements.result.textContent, /No rules matched/);
});

test('page script displays a real prediction from the running Flask API', async () => {
  const page = setup(fetch);
  await page.submit();
  assert.equal(page.elements.result.className, 'show ok');
  const item = page.elements.result.children[0].children[0];
  assert.equal(item.children[0].textContent, 'Pump It Up (M02)');
  assert.equal(item.children[1].textContent, 'Confidence 86.75% | Lift 5.95 | Support 3.89%');
});
