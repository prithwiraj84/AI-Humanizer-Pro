// ==========================================
//   SHARED UI HELPERS (loaded on every page)
// ==========================================

const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

// SQLite stores timestamps in UTC as "YYYY-MM-DD HH:MM:SS"
const parseUtc = (timestamp) => new Date(String(timestamp).replace(' ', 'T') + 'Z');

const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function icon(name, cls = '') {
    return `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

function toast(message, type = 'success') {
    const stack = document.getElementById('toastStack');
    if (!stack) return;
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.setAttribute('role', type === 'error' ? 'alert' : 'status');
    el.innerHTML = `${icon(type === 'error' ? 'alert' : 'check-circle')}<span>${escapeHtml(message)}</span>`;
    stack.appendChild(el);
    setTimeout(() => {
        el.classList.add('is-leaving');
        el.addEventListener('animationend', () => el.remove(), { once: true });
    }, 3200);
}

// Animate a number from its previous value to `to`
function countUp(el, to, { decimals = 0, suffix = '', duration = 900 } = {}) {
    const from = parseFloat(el.dataset.value) || 0;
    el.dataset.value = to;
    const start = performance.now();
    const step = (now) => {
        const t = prefersReducedMotion() ? 1 : Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = (from + (to - from) * eased).toFixed(decimals) + suffix;
        if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

async function copyText(text, button) {
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        // Clipboard API is unavailable on plain-HTTP origins; fall back to a hidden textarea
        const area = document.createElement('textarea');
        area.value = text;
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        area.remove();
    }
    if (button && !button.dataset.copying) {
        const original = button.innerHTML;
        button.dataset.copying = '1';
        button.innerHTML = `${icon('check')}<span>Copied</span>`;
        setTimeout(() => {
            button.innerHTML = original;
            delete button.dataset.copying;
        }, 1600);
    }
}

// Light syntax colouring for <pre data-lang="json|code"> blocks (strings, keys, numbers)
function highlightCode(pre) {
    const src = pre.textContent;
    const isJson = pre.dataset.lang === 'json';
    const pattern = isJson
        ? /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\b\d+(?:\.\d+)?\b/g
        : /("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(true|false|True|False|null)\b/g;
    let html = '';
    let last = 0;
    let match;
    while ((match = pattern.exec(src))) {
        html += escapeHtml(src.slice(last, match.index));
        if (isJson && match[1]) {
            html += `<span class="${match[2] ? 'tok-key' : 'tok-str'}">${escapeHtml(match[1])}</span>${match[2] ? escapeHtml(match[2]) : ''}`;
        } else if (match[1]) {
            html += `<span class="tok-str">${escapeHtml(match[1])}</span>`;
        } else if (match[isJson ? 3 : 2]) {
            html += `<span class="tok-bool">${escapeHtml(match[0])}</span>`;
        } else {
            html += `<span class="tok-num">${escapeHtml(match[0])}</span>`;
        }
        last = pattern.lastIndex;
    }
    pre.innerHTML = html + escapeHtml(src.slice(last));
}

// Tabbed code blocks: <div data-tabs> with .code-tab[data-tab] buttons and [data-panel] panes
function initCodeTabs(root = document) {
    root.querySelectorAll('[data-tabs]').forEach(group => {
        const tabs = group.querySelectorAll('.code-tab');
        const panels = group.querySelectorAll('[data-panel]');
        tabs.forEach(tab => tab.addEventListener('click', () => {
            tabs.forEach(t => {
                t.classList.toggle('active', t === tab);
                t.setAttribute('aria-selected', t === tab);
            });
            panels.forEach(p => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
        }));
    });
}

// "On this page" sidebar: highlights the section in view and scrolls smoothly to clicked sections
function initScrollspy(linkSelector = '.docs-nav a', sectionSelector = '.doc-section') {
    const links = document.querySelectorAll(linkSelector);
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            links.forEach(link => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`));
        });
    }, { rootMargin: '-90px 0px -65% 0px' });
    document.querySelectorAll(sectionSelector).forEach(section => observer.observe(section));

    links.forEach(link => link.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelector(link.getAttribute('href'))
            .scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
        history.replaceState(null, '', link.getAttribute('href'));
    }));
}

// Copies the visible <pre> inside the .code block that holds the button
function copyCode(button) {
    const block = button.closest('.code');
    const pre = [...block.querySelectorAll('pre')].find(p => !p.hidden);
    copyText(pre.textContent, button);
}

// ==========================================
//   PAYMENTS (Razorpay Checkout, loaded only when someone buys)
// ==========================================

let razorpayScript = null;

function loadRazorpay() {
    razorpayScript = razorpayScript || new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://checkout.razorpay.com/v1/checkout.js';
        script.onload = resolve;
        script.onerror = () => {
            razorpayScript = null;
            reject(new Error('Could not load the payment window. Check your connection and try again.'));
        };
        document.head.appendChild(script);
    });
    return razorpayScript;
}

// productId: 'pro', 'pro_yearly', 'topup', ... `autopay` starts a Razorpay Subscription (auto-renewal)
// instead of a one-time order; `coupon` applies to one-time orders only.
async function buyPlan(productId, button, { coupon = '', autopay = false } = {}) {
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span><span>Opening checkout…</span>';
    try {
        const response = await fetch(autopay ? '/api/billing/subscribe' : '/api/billing/order', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(autopay ? { product: productId } : { product: productId, coupon })
        });
        if (response.status === 401) {
            window.location.href = '/signup?next=/pricing';
            return;
        }
        const order = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(order.error || 'Could not start checkout. Please try again.');
        if (order.free) {
            // A 100% coupon: nothing to pay, already granted
            window.location.href = '/dashboard?paid=1';
            return;
        }

        await loadRazorpay();
        // Razorpay's window covers the page and reports back through these callbacks, so the button
        // is restored right after opening (Razorpay can also fail with its own alert and no callback)
        const options = {
            key: order.key_id,
            name: 'AI Humanizer',
            description: order.description,
            prefill: order.prefill,
            theme: { color: '#16171B' },
        };
        const checkout = new Razorpay(autopay
            ? { ...options, subscription_id: order.subscription_id, handler: (p) => confirmPayment(p, '/api/billing/verify-subscription') }
            : { ...options, order_id: order.order_id, amount: order.amount, currency: order.currency, handler: (p) => confirmPayment(p) });
        checkout.on('payment.failed', (failure) => {
            toast((failure.error && failure.error.description) || 'The payment failed. You have not been charged.', 'error');
        });
        checkout.open();
    } catch (e) {
        toast(e.message, 'error');
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
}

async function confirmPayment(payment, endpoint = '/api/billing/verify') {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payment)
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'We could not confirm the payment yet.');
        window.location.href = '/dashboard?paid=1';
    } catch (e) {
        toast(`${e.message} If you were charged, your plan will activate within a few minutes.`, 'error');
    }
}

// ==========================================
//   PRICING CARDS (pricing page and landing page)
// ==========================================

const pricingState = { cycle: 'monthly', coupon: null };

function formatMoney(amount, currency) {
    const value = amount / 100;
    const digits = Number.isInteger(value) ? 0 : 2;
    const symbol = { INR: '₹', USD: '$' }[currency] || `${currency} `;
    return symbol + value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// Same rule as the server: whole-unit discount rounded down, never below the ₹1 / $1 minimum charge
function discountedAmount(amount, percent) {
    let result = amount - Math.floor(amount * percent / 100);
    if (result > 0 && result < 100) result = Math.min(amount, 100);
    return result;
}

function productFor(card) {
    const plan = card.dataset.plan;
    return plan === 'topup' || pricingState.cycle === 'monthly' ? plan : `${plan}_${pricingState.cycle}`;
}

function renderPricing() {
    const root = document.getElementById('pricingCards');
    if (!root) return;
    const { cycle, coupon } = pricingState;
    root.querySelectorAll('[data-monthly]').forEach(el => {
        el.textContent = el.dataset[cycle] || ' ';
    });
    root.querySelectorAll('[data-amount-monthly]').forEach(card => {
        const amount = Number(card.dataset[cycle === 'monthly' ? 'amountMonthly' : 'amountYearly']);
        const was = card.querySelector('.price-was');
        const price = card.querySelector('.amount');
        const applies = coupon && coupon.products.includes(productFor(card)) && !autopayOn(card);
        const discounted = applies ? discountedAmount(amount, coupon.percent_off) : amount;
        was.hidden = discounted === amount;
        was.textContent = formatMoney(amount, root.dataset.currency);
        price.textContent = formatMoney(discounted, root.dataset.currency);
    });
}

function autopayOn(card) {
    const toggle = document.getElementById('autopayToggle');
    return Boolean(toggle && toggle.checked && card.dataset.plan !== 'topup');
}

function buyFromCard(button) {
    const card = button.closest('[data-plan]');
    const autopay = autopayOn(card);
    buyPlan(productFor(card), button, { coupon: autopay || !pricingState.coupon ? '' : pricingState.coupon.code, autopay });
}

(() => {
    const root = document.getElementById('pricingCards');
    if (!root) return;
    initSegmented(document.getElementById('cycleControl'), (cycle) => {
        pricingState.cycle = cycle;
        renderPricing();
    });
    const autopay = document.getElementById('autopayToggle');
    if (autopay) autopay.addEventListener('change', renderPricing);

    const form = document.getElementById('couponForm');
    if (!form) return;
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const input = document.getElementById('couponCode');
        const status = document.getElementById('couponStatus');
        const button = document.getElementById('couponBtn');
        const code = input.value.trim().toUpperCase();
        if (!code) {
            pricingState.coupon = null;
            status.textContent = '';
            renderPricing();
            return;
        }
        button.disabled = true;
        try {
            const response = await fetch('/api/billing/coupon', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ coupon: code })
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.error || 'That coupon code isn\'t valid.');
            pricingState.coupon = data;
            input.value = data.code;
            status.textContent = `${data.code}: ${data.percent_off}% off ${data.summary}.` +
                (autopay && autopay.checked ? ' Coupons apply to one-time payments, not auto-renewal.' : '');
            status.classList.remove('is-error');
        } catch (error) {
            pricingState.coupon = null;
            status.textContent = error.message;
            status.classList.add('is-error');
        } finally {
            button.disabled = false;
            renderPricing();
        }
    });
})();

// Segmented control: slide the thumb under the checked option
function initSegmented(control, onChange) {
    const thumb = control.querySelector('.segmented-thumb');
    const move = () => {
        const label = control.querySelector('input:checked + label');
        if (!label || !thumb) return;
        thumb.style.width = `${label.offsetWidth}px`;
        thumb.style.transform = `translateX(${label.offsetLeft}px)`;
    };
    control.addEventListener('change', () => {
        move();
        if (onChange) onChange(control.querySelector('input:checked').value);
    });
    window.addEventListener('resize', move);
    if (thumb) {
        thumb.style.transition = 'none';
        move();
        requestAnimationFrame(() => { thumb.style.transition = ''; });
    }
    if (document.fonts) document.fonts.ready.then(move);
    return move;
}

// Sticky nav gets a hairline border once the page scrolls
(() => {
    const nav = document.querySelector('.nav');
    if (!nav) return;
    const onScroll = () => nav.classList.toggle('is-scrolled', window.scrollY > 4);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
})();
