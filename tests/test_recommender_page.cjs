// Execute the page script with a minimal DOM adapter. This checks behavior,
// including a live API request, without claiming full browser/visual coverage.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const test = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../frontend/ecommerce_recommender.html'), 'utf8');
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
  for (const id of ['predictForm', 'api_base', 'cart', 'top_n', 'result', 'submitBtn']) elements[id] = element();
  elements.top_n.value = '3';
  elements.cart.value = 'H06, H07';
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

test('blank and comma-only carts show a clear error before any request', async () => {
  for (const value of ['', '   ', ', ,']) {
    const page = setup();
    page.elements.cart.value = value;
    await page.submit();
    assert.match(page.elements.result.textContent, /at least one product SKU/);
    assert.equal(page.calls.length, 0);
    assert.equal(page.elements.cart.focused, true);
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

test('deduplicates SKUs and renders product names as text', async () => {
  const page = setup(async () => ({ok: true, json: async () => ({
    model: 'Apriori', recommendations: [{sku: 'H03', product_name: '<b>Product</b>', confidence: .8, lift: 2}],
  })}));
  page.elements.cart.value = ' H06, H07, H06 ';
  await page.submit();
  assert.deepEqual(JSON.parse(page.calls[0][1].body), {cart: ['H06', 'H07'], top_n: 3});
  const item = page.elements.result.children[0].children[0];
  assert.equal(item.children[0].textContent, '<b>Product</b> (H03)');
  assert.equal(page.elements.submitBtn.disabled, false);
});

test('server validation errors and empty results are readable', async () => {
  const failure = setup(async () => ({ok: false, json: async () => ({error: 'Unknown product SKUs', unknown_skus: ['BAD']})}));
  await failure.submit();
  assert.equal(failure.elements.result.textContent, 'Unknown product SKUs: BAD');
  const empty = setup();
  await empty.submit();
  assert.match(empty.elements.result.textContent, /No rules matched/);
});

test('page script displays a real prediction from the running Flask API', async () => {
  const page = setup(fetch);
  await page.submit();
  assert.equal(page.elements.result.className, 'show ok');
  const item = page.elements.result.children[0].children[0];
  assert.equal(item.children[0].textContent, 'Electric Kettle (H03)');
  assert.equal(item.children[1].textContent, 'Confidence 93.85% | Lift 12.72');
});
