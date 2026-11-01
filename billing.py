"""
API plans, top-ups, coupons, referrals, regional pricing and Razorpay payments.

What can be bought
  - Plans (Starter, Pro, Business) for 30 days, or for a year at the price of 10 months.
    Each purchase adds the plan's requests; unused requests and remaining days carry over
    while a plan is active.
  - Top-ups: extra requests that never expire. They're used once the plan's requests run out
    (or the plan ends), so occasional users don't need a plan at all.
New accounts get a small one-time trial so developers can test their integration. The web
editor stays free.

Prices follow the visitor's country (from their IP address, looked up in a local copy of
DB-IP's free database), grouped by World Bank income level so the API stays affordable
everywhere. India is billed in rupees; other regions in US dollars, which needs
"International Payments" enabled on the Razorpay account. Razorpay's minimum charge is
₹1 / $1, so no price or discount goes below that.

Payment flows
  - One-time: create an order on the server -> Razorpay Checkout in the browser -> verify the
    payment signature on the server. The `order.paid` webhook repeats the confirmation in case
    the browser closes first; fulfilment is idempotent, so a purchase is granted exactly once.
  - Auto-renewal (optional, rupee plans only, RAZORPAY_AUTOPAY=1): a Razorpay Subscription
    charges the card or UPI AutoPay mandate every cycle, and each `subscription.charged`
    webhook grants one cycle. Needs Subscriptions enabled on the Razorpay account.

After a purchase is committed, the functions in `on_paid` run (the app uses them to issue the
invoice and email the receipt).
"""
import hashlib
import hmac
import logging
import os
import secrets
from datetime import timedelta
from typing import NamedTuple

import requests

import db

logger = logging.getLogger(__name__)

TRIAL_PLAN = 'trial'
NO_PLAN = 'none'              # account row that only holds top-up credits
TRIAL_PERIOD = timedelta(days=3650)
AUTOPAY_GRACE = timedelta(days=2)    # auto-renewal charges can land a little after the period ends
REMINDER_WINDOW = timedelta(days=3)  # email a reminder this long before a plan ends
QUOTA_WARNING_PERCENT = 80          # email a warning when this share of the requests is used
MIN_CHARGE = 100                     # Razorpay's minimum: ₹1.00 / $1.00, in paise / cents

# Requests per 30 days. "best" marks the plan highlighted as best value.
PLANS = {
    'starter': {'name': 'Starter', 'requests': 5_000, 'blurb': 'For side projects and prototypes.'},
    'pro': {'name': 'Pro', 'requests': 30_000, 'blurb': 'For apps in production.', 'best': True},
    'business': {'name': 'Business', 'requests': 150_000, 'blurb': 'For products with steady traffic.'},
}

# A yearly plan gives 12 months of requests for the price of 10
CYCLES = {
    'monthly': {'label': '30 days', 'period': timedelta(days=30), 'months': 1, 'price_months': 1,
                'razorpay_period': 'monthly', 'autopay_count': 60},
    'yearly': {'label': '1 year', 'period': timedelta(days=365), 'months': 12, 'price_months': 10,
               'razorpay_period': 'yearly', 'autopay_count': 5},
}
PLAN_PERIOD = CYCLES['monthly']['period']

TOPUP_ID = 'topup'
TOPUP_REQUESTS = 2_000

# Referral program: credits for both sides when a referred account makes its first purchase
REFERRER_CREDITS = 1_000
REFERRED_CREDITS = 500
REFERRAL_REWARD_LIMIT = 50   # rewarded referrals per referrer
CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'   # no 0/O or 1/I lookalikes

# Prices in the currency's smallest unit (paise / cents): a 30-day plan, or one top-up
PRICE_TIERS = {
    'india': {'currency': 'INR', 'prices': {'starter': 9900, 'pro': 29900, 'business': 99900, TOPUP_ID: 5900}},
    'lower': {'currency': 'USD', 'prices': {'starter': 100, 'pro': 299, 'business': 899, TOPUP_ID: 100}},
    'middle': {'currency': 'USD', 'prices': {'starter': 199, 'pro': 499, 'business': 1499, TOPUP_ID: 149}},
    'high': {'currency': 'USD', 'prices': {'starter': 299, 'pro': 799, 'business': 2499, TOPUP_ID: 199}},
}
CURRENCY_SYMBOLS = {'INR': '₹', 'USD': '$'}


def _build_products():
    products = {}
    for plan_id, plan in PLANS.items():
        for cycle_id, cycle in CYCLES.items():
            product_id = plan_id if cycle_id == 'monthly' else f'{plan_id}_{cycle_id}'
            requests_ = plan['requests'] * cycle['months']
            products[product_id] = {
                'id': product_id, 'kind': 'plan', 'plan': plan_id, 'cycle': cycle_id, 'requests': requests_,
                'name': f"{plan['name']} plan, {cycle['label']}",
                'description': f"{plan['name']} API plan: {requests_:,} requests, {cycle['label']}",
            }
    products[TOPUP_ID] = {
        'id': TOPUP_ID, 'kind': 'topup', 'plan': None, 'cycle': None, 'requests': TOPUP_REQUESTS,
        'name': f'Top-up, {TOPUP_REQUESTS:,} requests',
        'description': f'API top-up: {TOPUP_REQUESTS:,} requests (no expiry)',
    }
    return products


# Everything that can be bought, by id: 'pro' (30 days), 'pro_yearly', 'topup', ...
PRODUCTS = _build_products()

# Country groups based on World Bank income classifications (2024); unlisted countries pay "middle".
HIGH_INCOME = set('''
AD AE AG AT AU AW BB BE BG BH BM BN BS CA CH CL CW CY CZ DE DK EE ES FI FO FR GB GI GL GR GU GY HK HR HU
IE IL IM IS IT JP KN KR KW KY LI LT LU LV MC MF MO MP MT NC NL NO NR NZ OM PA PF PL PR PT PW QA RO RU SA
SC SE SG SI SK SM SX TC TT TW US UY VG VI
'''.split())
LOWER_INCOME = set('''
AF AO BD BF BI BJ BO BT CD CF CG CI CM CV DJ EG ER ET FM GH GM GN GW HN HT IR KE KG KH KI KM KP LA LB LK
LR LS MA MG ML MM MR MW MZ NE NG NI NP PG PH PK PS RW SB SD SL SN SO SS ST SY SZ TD TG TJ TL TN TZ UG UZ
VN VU WS YE ZM ZW
'''.split())

GEOIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'geoip', 'dbip-country-lite.mmdb')
RAZORPAY_API = 'https://api.razorpay.com/v1'

# Auto-renewal states in which the customer is (or is about to be) charged automatically
AUTOPAY_LIVE = ('authenticated', 'active', 'pending')

_geoip = None
_geoip_loaded = False

# Called as hook(payment_id, extras) after a purchase is committed; failures are logged, never raised
on_paid = []


def config(name, default=''):
    # Read at call time so .env (loaded by app.py) and test overrides apply
    return os.environ.get(name, default)


def trial_requests():
    return int(config('API_TRIAL_REQUESTS', '50'))


def is_enabled():
    return bool(config('RAZORPAY_KEY_ID') and config('RAZORPAY_KEY_SECRET'))


def autopay_enabled():
    return is_enabled() and config('RAZORPAY_AUTOPAY') == '1'

# ==========================================
#   REGIONAL PRICING
# ==========================================

def lookup_country(ip):
    """(ISO code, English name) for an IP address, or (None, None) if unknown."""
    global _geoip, _geoip_loaded
    if not _geoip_loaded:
        _geoip_loaded = True
        try:
            import maxminddb
            _geoip = maxminddb.open_database(GEOIP_PATH)   # memory-mapped: little RAM
        except Exception as e:
            logger.warning('GeoIP database unavailable (%s); run scripts/fetch_geoip.py', type(e).__name__)
    if _geoip is None or not ip:
        return None, None
    try:
        country = (_geoip.get(ip) or {}).get('country') or {}
    except ValueError:
        return None, None
    return country.get('iso_code'), (country.get('names') or {}).get('en')


def tier_for_country(code):
    if code == 'IN':
        return 'india'
    if code in HIGH_INCOME:
        return 'high'
    if code in LOWER_INCOME:
        return 'lower'
    return 'middle'


def format_price(amount, currency):
    symbol = CURRENCY_SYMBOLS.get(currency, currency + ' ')
    value = amount / 100
    return f'{symbol}{value:,.0f}' if value == int(value) else f'{symbol}{value:,.2f}'


def price_for(product_id, tier):
    product = PRODUCTS[product_id]
    prices = PRICE_TIERS[tier]['prices']
    if product['kind'] == 'topup':
        return prices[TOPUP_ID]
    return prices[product['plan']] * CYCLES[product['cycle']]['price_months']


def product_name(product_id):
    return PRODUCTS.get(product_id, {}).get('name', product_id)


def pricing_for(country_code, country_name=None):
    """Everything the pricing cards need for one visitor."""
    tier = tier_for_country(country_code)
    currency = PRICE_TIERS[tier]['currency']
    plans = []
    for plan_id, plan in PLANS.items():
        amount = price_for(plan_id, tier)
        yearly = price_for(f'{plan_id}_yearly', tier)
        # A yearly price shown per month: whole rupees, or cents
        per_month = round(yearly / 1200) * 100 if currency == 'INR' else round(yearly / 12)
        plans.append({
            'id': plan_id, 'name': plan['name'], 'requests': plan['requests'], 'blurb': plan['blurb'],
            'best': plan.get('best', False), 'amount': amount, 'currency': currency,
            'price': format_price(amount, currency),
            'per_1k': format_price(round(amount / plan['requests'] * 1000), currency) if amount * 1000 >= plan['requests'] else None,
            'yearly': {'id': f'{plan_id}_yearly', 'amount': yearly, 'price': format_price(yearly, currency),
                       'per_month': format_price(per_month, currency), 'requests': plan['requests'] * 12,
                       'savings': format_price(amount * 12 - yearly, currency)},
        })
    topup = price_for(TOPUP_ID, tier)
    return {'country': country_code, 'country_name': country_name, 'tier': tier, 'currency': currency,
            'plans': plans, 'enabled': is_enabled(), 'trial_requests': trial_requests(),
            'topup': {'id': TOPUP_ID, 'requests': TOPUP_REQUESTS, 'amount': topup, 'price': format_price(topup, currency)},
            'autopay': autopay_enabled() and currency == 'INR'}

# ==========================================
#   COUPONS
# ==========================================

class PaymentError(Exception):
    """A payment step failed; the message is safe to show to the user."""


class CouponError(PaymentError):
    """A coupon can't be used; the message is safe to show to the user."""


def normalize_code(code):
    return code.strip().upper() if isinstance(code, str) else ''


def coupon_applies(applies_to, product_id):
    """applies_to: comma-separated product ids and groups ('plans', 'monthly', 'yearly', 'topup'); empty = everything."""
    targets = {t.strip() for t in (applies_to or '').split(',') if t.strip()}
    if not targets:
        return True
    product = PRODUCTS[product_id]
    return bool(targets & {product_id, product['cycle'], 'plans' if product['kind'] == 'plan' else TOPUP_ID})


def check_coupon(conn, code, user_id, product_id, now):
    """The coupon row if `code` can be used for this purchase, otherwise CouponError."""
    coupon = conn.execute('SELECT * FROM coupons WHERE code = ?', (code,)).fetchone()
    if coupon is None or not coupon['active']:
        raise CouponError("That coupon code isn't valid.")
    if coupon['expires_at'] and db.to_datetime(coupon['expires_at']) <= now:
        raise CouponError('That coupon has expired.')
    if coupon['max_redemptions'] is not None and coupon['redeemed_count'] >= coupon['max_redemptions']:
        raise CouponError('That coupon has been fully redeemed.')
    if not coupon_applies(coupon['applies_to'], product_id):
        raise CouponError("That coupon doesn't apply to this purchase.")
    if user_id is not None and conn.execute(
            "SELECT 1 FROM payments WHERE user_id = ? AND coupon_code = ? AND status IN ('paid', 'refunded')",
            (user_id, code)).fetchone():
        raise CouponError("You've already used that coupon.")
    return coupon


def coupon_info(code, user_id, now):
    """What a coupon gives this user, for showing discounts on the pricing page: percent off and the
    product ids it can be used for. CouponError if it can't be used for anything."""
    code = normalize_code(code)
    if not code:
        raise CouponError('Enter a coupon code.')
    applies, error, coupon = [], None, None
    conn = db.connect()
    try:
        for product_id in PRODUCTS:
            try:
                coupon = check_coupon(conn, code, user_id, product_id, now)
                applies.append(product_id)
            except CouponError as e:
                error = error or e
    finally:
        conn.close()
    if not applies:
        raise error

    plans = [p for p in PLANS if any(PRODUCTS[a]['plan'] == p for a in applies)]
    cycles = {PRODUCTS[a]['cycle'] for a in applies if PRODUCTS[a]['kind'] == 'plan'}
    parts = []
    if plans:
        names = 'all plans' if len(plans) == len(PLANS) else \
            ' and '.join(filter(None, [', '.join(PLANS[p]['name'] for p in plans[:-1]), PLANS[plans[-1]]['name']])) + ' plans'
        parts.append(names + ('' if len(cycles) > 1 else f" ({CYCLES[cycles.pop()]['label']})"))
    if TOPUP_ID in applies:
        parts.append('top-ups')
    return {'code': code, 'percent_off': coupon['percent_off'], 'products': applies, 'summary': ' and '.join(parts)}


def quote(product_id, country_code, coupon_code=None, user_id=None, now=None):
    """The price of a product for this visitor, after any coupon."""
    if product_id not in PRODUCTS:
        raise PaymentError('Unknown plan.')
    tier = tier_for_country(country_code)
    currency = PRICE_TIERS[tier]['currency']
    original = price_for(product_id, tier)
    result = {'product': product_id, 'name': PRODUCTS[product_id]['name'], 'currency': currency,
              'original': original, 'amount': original, 'discount': 0, 'coupon': None, 'percent_off': 0}
    code = normalize_code(coupon_code)
    if code:
        conn = db.connect()
        try:
            coupon = check_coupon(conn, code, user_id, product_id, now or db.utcnow())
        finally:
            conn.close()
        amount = original - original * coupon['percent_off'] // 100
        if 0 < amount < MIN_CHARGE:
            amount = min(original, MIN_CHARGE)
        if amount == original:
            raise CouponError("That coupon can't lower this price any further.")
        result.update(amount=amount, discount=original - amount, coupon=code, percent_off=coupon['percent_off'])
    result['price'] = format_price(result['amount'], currency)
    result['original_price'] = format_price(original, currency)
    return result

# ==========================================
#   RAZORPAY
# ==========================================

def razorpay_request(method, path, payload=None):
    try:
        response = requests.request(method, RAZORPAY_API + path, json=payload, timeout=15,
                                    auth=(config('RAZORPAY_KEY_ID'), config('RAZORPAY_KEY_SECRET')))
    except requests.RequestException as e:
        logger.error('Razorpay %s %s failed: %s', method, path, type(e).__name__)
        raise PaymentError('The payment service is not responding. Please try again.')
    if response.status_code >= 400:
        try:
            description = response.json().get('error', {}).get('description')
        except ValueError:
            description = None
        logger.error('Razorpay %s %s returned %s: %s', method, path, response.status_code, description)
        raise PaymentError(description or 'The payment could not be processed.')
    return response.json()


def _signature_matches(message, signature):
    expected = hmac.new(config('RAZORPAY_KEY_SECRET').encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or '')


def verify_payment_signature(order_id, payment_id, signature):
    return _signature_matches(f'{order_id}|{payment_id}', signature)


def verify_subscription_signature(payment_id, subscription_id, signature):
    # Razorpay signs subscription payments as payment_id|subscription_id (the reverse of orders)
    return _signature_matches(f'{payment_id}|{subscription_id}', signature)


def verify_webhook_signature(body, signature):
    secret = config('RAZORPAY_WEBHOOK_SECRET')
    if not secret:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or '')


def create_order(user_id, product_id, country_code, now, coupon_code=None):
    """Create a Razorpay order at the visitor's regional price (after any coupon) and record it.
    A coupon that makes the purchase free is granted straight away, without Razorpay."""
    price = quote(product_id, country_code, coupon_code, user_id, now)
    currency, amount = price['currency'], price['amount']
    summary = {'amount': amount, 'currency': currency, 'key_id': config('RAZORPAY_KEY_ID'), 'product': product_id,
               'plan': price['name'], 'price': price['price'], 'original_price': price['original_price'],
               'discount': price['discount'], 'coupon': price['coupon']}

    if amount == 0:
        order_id = f'free_{secrets.token_hex(12)}'
    else:
        notes = {'user_id': str(user_id), 'product': product_id, 'country': country_code or 'unknown'}
        if price['coupon']:
            notes['coupon'] = price['coupon']
        order_id = razorpay_request('POST', '/orders', {
            'amount': amount, 'currency': currency, 'receipt': f'u{user_id}-{int(now.timestamp())}', 'notes': notes,
        })['id']

    conn = db.connect()
    try:
        conn.execute('''INSERT INTO payments (user_id, plan_id, country, currency, amount, razorpay_order_id, status,
                                              created_at, coupon_code, discount)
                        VALUES (?, ?, ?, ?, ?, ?, 'created', ?, ?, ?)''',
                     (user_id, product_id, country_code, currency, amount, order_id, now, price['coupon'], price['discount']))
        conn.commit()
    finally:
        conn.close()

    if amount == 0:
        fulfil_payment(order_id, None, now, 0, currency)
        return {**summary, 'free': True}
    return {**summary, 'order_id': order_id}


def capture_if_needed(payment_id, amount, currency):
    """Payments are captured automatically on most accounts; capture here if this one isn't."""
    payment = razorpay_request('GET', f'/payments/{payment_id}')
    if payment.get('status') == 'authorized':
        payment = razorpay_request('POST', f'/payments/{payment_id}/capture', {'amount': amount, 'currency': currency})
    return payment


def fulfil_payment(order_id, payment_id, now, amount=None, currency=None):
    """Mark an order paid and grant what was bought, exactly once. Returns the payment row, or None."""
    conn = db.connect()
    extras = None
    try:
        payment = conn.execute('SELECT * FROM payments WHERE razorpay_order_id = ?', (order_id,)).fetchone()
        if payment is None:
            logger.warning('Payment for unknown order %s', order_id)
            return None
        if amount is not None and (amount != payment['amount'] or currency != payment['currency']):
            logger.error('Order %s paid %s %s, expected %s %s', order_id, amount, currency, payment['amount'], payment['currency'])
            return None
        updated = conn.execute('''UPDATE payments SET status = 'paid', razorpay_payment_id = ?, paid_at = ?
                                  WHERE razorpay_order_id = ? AND status = 'created' ''', (payment_id, now, order_id))
        if updated.rowcount == 1:
            extras = _apply_purchase(conn, payment, now)
            logger.info('Order %s paid: %s for user %s', order_id, payment['plan_id'], payment['user_id'])
        conn.commit()
    finally:
        conn.close()
    if extras is not None:
        _run_paid_hooks(payment['id'], extras)
    return payment


def _apply_purchase(conn, payment, now, grace=timedelta(0)):
    """Grant a paid purchase and its side effects (coupon count, referral reward), inside the caller's transaction."""
    grant_product(conn, payment['user_id'], payment['plan_id'], now, grace)
    if payment['coupon_code']:
        conn.execute('UPDATE coupons SET redeemed_count = redeemed_count + 1 WHERE code = ?', (payment['coupon_code'],))
    reward = reward_referral(conn, payment['user_id'], now) if payment['amount'] > 0 else None
    return {'referral_reward': reward}


def _run_paid_hooks(payment_id, extras):
    for hook in on_paid:
        try:
            hook(payment_id, extras)
        except Exception as e:
            logger.error('After-payment step %s failed: %s', getattr(hook, '__name__', hook), type(e).__name__)

# ==========================================
#   AUTO-RENEWAL (RAZORPAY SUBSCRIPTIONS)
# ==========================================

def razorpay_plan_for(product_id, currency, amount):
    """Razorpay's plan object for a product at a price, created once and cached."""
    conn = db.connect()
    try:
        row = conn.execute('SELECT razorpay_plan_id FROM razorpay_plans WHERE product_id = ? AND currency = ? AND amount = ?',
                           (product_id, currency, amount)).fetchone()
        if row:
            return row['razorpay_plan_id']
    finally:
        conn.close()

    product = PRODUCTS[product_id]
    plan = razorpay_request('POST', '/plans', {
        'period': CYCLES[product['cycle']]['razorpay_period'], 'interval': 1,
        'item': {'name': f"{config('BUSINESS_NAME', 'AI Humanizer')}: {product['name']}", 'amount': amount,
                 'currency': currency, 'description': product['description']},
        'notes': {'product': product_id},
    })
    conn = db.connect()
    try:
        conn.execute('''INSERT INTO razorpay_plans (product_id, currency, amount, razorpay_plan_id) VALUES (?, ?, ?, ?)
                        ON CONFLICT DO NOTHING''', (product_id, currency, amount, plan['id']))
        conn.commit()
        # Another request may have created one at the same moment; everyone uses the stored one
        return conn.execute('SELECT razorpay_plan_id FROM razorpay_plans WHERE product_id = ? AND currency = ? AND amount = ?',
                            (product_id, currency, amount)).fetchone()['razorpay_plan_id']
    finally:
        conn.close()


def live_autopay(conn, user_id):
    return conn.execute(f'''SELECT * FROM autopay_subscriptions WHERE user_id = ? AND status IN ({",".join("?" * len(AUTOPAY_LIVE))})
                            ORDER BY created_at DESC LIMIT 1''', (user_id, *AUTOPAY_LIVE)).fetchone()


def create_autopay(user_id, product_id, country_code, now):
    """Start a Razorpay Subscription for a plan; Checkout then sets up the card or UPI AutoPay mandate."""
    product = PRODUCTS.get(product_id)
    if not product or product['kind'] != 'plan':
        raise PaymentError('Auto-renewal is available for plans only.')
    tier = tier_for_country(country_code)
    currency = PRICE_TIERS[tier]['currency']
    if currency != 'INR':
        raise PaymentError('Auto-renewal is available for payments in rupees only.')

    conn = db.connect()
    try:
        if live_autopay(conn, user_id):
            raise PaymentError('Auto-renewal is already on. Turn it off first to switch plans.')
    finally:
        conn.close()

    amount = price_for(product_id, tier)
    plan_id = razorpay_plan_for(product_id, currency, amount)
    subscription = razorpay_request('POST', '/subscriptions', {
        'plan_id': plan_id, 'total_count': CYCLES[product['cycle']]['autopay_count'], 'quantity': 1, 'customer_notify': 1,
        'notes': {'user_id': str(user_id), 'product': product_id},
    })
    conn = db.connect()
    try:
        conn.execute('''INSERT INTO autopay_subscriptions (user_id, product_id, country, currency, amount, razorpay_subscription_id,
                                                           razorpay_plan_id, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'created', ?, ?)''',
                     (user_id, product_id, country_code, currency, amount, subscription['id'], plan_id, now, now))
        conn.commit()
    finally:
        conn.close()
    return {'subscription_id': subscription['id'], 'key_id': config('RAZORPAY_KEY_ID'), 'amount': amount, 'currency': currency,
            'product': product_id, 'plan': product['name'], 'price': format_price(amount, currency)}


def record_subscription_charge(subscription_id, payment, now):
    """Record one auto-renewal charge and grant its cycle, exactly once per Razorpay payment."""
    conn = db.connect()
    payment_row = None
    try:
        sub = conn.execute('SELECT * FROM autopay_subscriptions WHERE razorpay_subscription_id = ?', (subscription_id,)).fetchone()
        if sub is None:
            logger.warning('Charge for unknown subscription %s', subscription_id)
            return None
        if payment.get('amount') != sub['amount'] or payment.get('currency') != sub['currency']:
            logger.error('Subscription %s charged %s %s, expected %s %s', subscription_id, payment.get('amount'),
                         payment.get('currency'), sub['amount'], sub['currency'])
            return None
        inserted = conn.execute('''INSERT INTO payments (user_id, plan_id, country, currency, amount, razorpay_order_id,
                                                         razorpay_payment_id, razorpay_subscription_id, status, created_at, paid_at)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'paid', ?, ?) ON CONFLICT DO NOTHING RETURNING id''',
                                (sub['user_id'], sub['product_id'], sub['country'], sub['currency'], sub['amount'],
                                 payment.get('order_id') or f"sub_{payment['id']}", payment['id'], subscription_id, now, now)).fetchone()
        if inserted is None:
            conn.commit()
            return None   # this charge was already recorded
        payment_row = conn.execute('SELECT * FROM payments WHERE id = ?', (inserted['id'],)).fetchone()
        extras = _apply_purchase(conn, payment_row, now, grace=AUTOPAY_GRACE)
        conn.execute('''UPDATE autopay_subscriptions SET status = 'active', updated_at = ?
                        WHERE id = ? AND status NOT IN ('cancelled', 'completed')''', (now, sub['id']))
        conn.commit()
        logger.info('Subscription %s charged: %s for user %s', subscription_id, sub['product_id'], sub['user_id'])
    finally:
        conn.close()
    _run_paid_hooks(payment_row['id'], extras)
    return payment_row


def update_autopay_status(subscription_id, status, now):
    """Follow Razorpay's subscription state. Cancelled/completed are final."""
    conn = db.connect()
    try:
        conn.execute('''UPDATE autopay_subscriptions SET status = ?, updated_at = ?
                        WHERE razorpay_subscription_id = ? AND status NOT IN ('cancelled', 'completed')''',
                     (status, now, subscription_id))
        conn.commit()
    finally:
        conn.close()


def cancel_autopay(user_id, now):
    """Stop future charges. What's already paid for stays active until the period ends."""
    conn = db.connect()
    try:
        sub = live_autopay(conn, user_id)
    finally:
        conn.close()
    if sub is None:
        raise PaymentError("Auto-renewal isn't on.")
    razorpay_request('POST', f"/subscriptions/{sub['razorpay_subscription_id']}/cancel", {'cancel_at_cycle_end': 0})
    update_autopay_status(sub['razorpay_subscription_id'], 'cancelled', now)

# ==========================================
#   PLANS, CREDITS & QUOTA
# ==========================================

def ensure_row(conn, user_id, now):
    """Every account that uses the API gets a row: the trial, or an empty one that can hold top-up credits."""
    if trial_requests() > 0:
        conn.execute('''INSERT INTO api_subscriptions (user_id, plan_id, requests_limit, requests_used, period_start, period_end)
                        VALUES (?, ?, ?, 0, ?, ?) ON CONFLICT (user_id) DO NOTHING''',
                     (user_id, TRIAL_PLAN, trial_requests(), now, now + TRIAL_PERIOD))
    else:
        conn.execute('''INSERT INTO api_subscriptions (user_id, plan_id, requests_limit, requests_used, period_start, period_end)
                        VALUES (?, ?, 0, 0, ?, ?) ON CONFLICT (user_id) DO NOTHING''', (user_id, NO_PLAN, now, now))


def grant_plan(conn, user_id, plan_id, now, cycle='monthly', grace=timedelta(0)):
    """Add a cycle's days and requests. Unused requests from an active plan or the trial carry over,
    and so do the remaining days of an active paid plan. `grace` is added for auto-renewals, whose
    charges may arrive a little late; it isn't counted twice when the next charge lands."""
    current = conn.execute('SELECT * FROM api_subscriptions WHERE user_id = ?', (user_id,)).fetchone()
    carry, leftover = 0, timedelta(0)
    if current is None:
        carry = trial_requests()   # trial not started yet: keep its free requests
    elif db.to_datetime(current['period_end']) > now:
        carry = max(0, current['requests_limit'] - current['requests_used'])
        if current['plan_id'] in PLANS:
            leftover = max(timedelta(0), db.to_datetime(current['period_end']) - now - grace)
    period = CYCLES[cycle]['period']
    conn.execute('''INSERT INTO api_subscriptions (user_id, plan_id, cycle, requests_limit, requests_used, period_start, period_end,
                                                   warned_80, reminder_sent)
                    VALUES (?, ?, ?, ?, 0, ?, ?, 0, 0)
                    ON CONFLICT (user_id) DO UPDATE SET plan_id = excluded.plan_id, cycle = excluded.cycle,
                        requests_limit = excluded.requests_limit, requests_used = 0, period_start = excluded.period_start,
                        period_end = excluded.period_end, warned_80 = 0, reminder_sent = 0''',
                 (user_id, plan_id, cycle, carry + PLANS[plan_id]['requests'] * CYCLES[cycle]['months'], now,
                  now + period + grace + leftover))


def add_credits(conn, user_id, credits, now):
    ensure_row(conn, user_id, now)
    conn.execute('UPDATE api_subscriptions SET credits = credits + ?, warned_80 = 0 WHERE user_id = ?', (credits, user_id))


def grant_product(conn, user_id, product_id, now, grace=timedelta(0)):
    product = PRODUCTS[product_id]
    if product['kind'] == 'topup':
        add_credits(conn, user_id, product['requests'], now)
    else:
        grant_plan(conn, user_id, product['plan'], now, product['cycle'], grace)


def revoke_product(conn, user_id, product_id, now):
    """Take back what a refunded purchase added, without going below what was already used."""
    product = PRODUCTS.get(product_id)
    row = conn.execute('SELECT * FROM api_subscriptions WHERE user_id = ?', (user_id,)).fetchone()
    if product is None or row is None:
        return
    if product['kind'] == 'topup':
        conn.execute('UPDATE api_subscriptions SET credits = CASE WHEN credits >= ? THEN credits - ? ELSE 0 END WHERE user_id = ?',
                     (product['requests'], product['requests'], user_id))
        return
    limit = max(row['requests_used'], row['requests_limit'] - product['requests'])
    period_end = max(now, db.to_datetime(row['period_end']) - CYCLES[product['cycle']]['period'])
    conn.execute('UPDATE api_subscriptions SET requests_limit = ?, period_end = ? WHERE user_id = ?', (limit, period_end, user_id))


class Charge(NamedTuple):
    """Outcome of consume_request. `warn`: this charge crossed the quota-warning threshold."""
    allowed: bool
    row: object
    from_plan: int = 0
    from_credits: int = 0
    warn: bool = False


def consume_request(user_id, now, n=1):
    """Use n API requests, the plan's first and then top-up credits; all or nothing."""
    conn = db.connect()
    try:
        ensure_row(conn, user_id, now)
        conn.commit()
        row = None
        for _ in range(5):
            row = conn.execute('SELECT * FROM api_subscriptions WHERE user_id = ?', (user_id,)).fetchone()
            plan_left = max(0, row['requests_limit'] - row['requests_used']) if db.to_datetime(row['period_end']) > now else 0
            from_plan = min(n, plan_left)
            from_credits = n - from_plan
            if from_credits > row['credits']:
                return Charge(False, row)
            # Compare-and-set: applies only if no other request used this account's requests since the read
            updated = conn.execute('''UPDATE api_subscriptions SET requests_used = requests_used + ?, credits = credits - ?
                                      WHERE user_id = ? AND requests_used = ? AND credits = ?''',
                                   (from_plan, from_credits, user_id, row['requests_used'], row['credits']))
            if updated.rowcount == 1:
                warn = _claim_quota_warning(conn, row, from_plan, from_credits)
                conn.commit()
                row = conn.execute('SELECT * FROM api_subscriptions WHERE user_id = ?', (user_id,)).fetchone()
                return Charge(True, row, from_plan, from_credits, warn)
            conn.commit()   # end the transaction so the next read sees the other request's change
        return Charge(False, row)
    finally:
        conn.close()


def _claim_quota_warning(conn, row, from_plan, from_credits):
    """True once per plan, when what's left (plan + credits) drops to 20% of the plan's requests."""
    limit = row['requests_limit']
    if not from_plan or row['plan_id'] == NO_PLAN or row['warned_80'] or limit <= 0:
        return False
    left = limit - row['requests_used'] - from_plan + row['credits'] - from_credits
    if left * 100 > limit * (100 - QUOTA_WARNING_PERCENT):   # integers: no floating-point edge at exactly 80%
        return False
    return conn.execute('UPDATE api_subscriptions SET warned_80 = 1 WHERE user_id = ? AND warned_80 = 0',
                        (row['user_id'],)).rowcount == 1


def refund_request(user_id, charge=None, count=None):
    """Give back requests that failed on our side: a whole charge, `count` of it, or one plan request."""
    from_plan, from_credits = (charge.from_plan, charge.from_credits) if charge else (1, 0)
    if count is not None:
        from_credits, from_plan = min(count, from_credits), min(max(0, count - from_credits), from_plan)
    conn = db.connect()
    try:
        conn.execute('''UPDATE api_subscriptions
                        SET requests_used = CASE WHEN requests_used >= ? THEN requests_used - ? ELSE 0 END, credits = credits + ?
                        WHERE user_id = ?''', (from_plan, from_plan, from_credits, user_id))
        conn.commit()
    finally:
        conn.close()


def claim_expiry_reminders(now, window=REMINDER_WINDOW):
    """Paid plans ending within `window` whose owner hasn't been reminded and isn't on auto-renewal.
    Each is marked as reminded, so a reminder goes out once per plan period."""
    conn = db.connect()
    try:
        rows = conn.execute(f'''
            SELECT s.*, u.username, u.email FROM api_subscriptions s JOIN users u ON u.id = s.user_id
            WHERE s.plan_id IN ({",".join("?" * len(PLANS))}) AND s.reminder_sent = 0 AND s.period_end > ? AND s.period_end <= ?
              AND NOT EXISTS (SELECT 1 FROM autopay_subscriptions a WHERE a.user_id = s.user_id
                              AND a.status IN ({",".join("?" * len(AUTOPAY_LIVE))}))''',
            (*PLANS, now, now + window, *AUTOPAY_LIVE)).fetchall()
        due = [row for row in rows if conn.execute(
            'UPDATE api_subscriptions SET reminder_sent = 1 WHERE user_id = ? AND reminder_sent = 0', (row['user_id'],)).rowcount == 1]
        conn.commit()
        return due
    finally:
        conn.close()


def plan_label(plan_id):
    if plan_id == TRIAL_PLAN:
        return 'Free trial'
    return PLANS.get(plan_id, {}).get('name', plan_id)

# ==========================================
#   REFERRALS
# ==========================================

def referral_code(conn, user_id):
    """The user's referral code, created on first use."""
    for _ in range(5):
        row = conn.execute('SELECT referral_code FROM users WHERE id = ?', (user_id,)).fetchone()
        if row is None or row['referral_code']:
            return row and row['referral_code']
        code = ''.join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        try:
            conn.execute('UPDATE users SET referral_code = ? WHERE id = ? AND referral_code IS NULL', (code, user_id))
            conn.commit()
        except db.IntegrityError:
            conn.rollback()   # code taken by someone else: try another
    return None


def record_referral(conn, user_id, code, now):
    """Link a new account to whoever referred it (inside the caller's transaction)."""
    code = normalize_code(code)
    if not code:
        return False
    referrer = conn.execute('SELECT id FROM users WHERE referral_code = ?', (code,)).fetchone()
    if referrer is None or referrer['id'] == user_id:
        return False
    conn.execute('UPDATE users SET referred_by = ? WHERE id = ? AND referred_by IS NULL', (referrer['id'], user_id))
    conn.execute('''INSERT INTO referrals (referrer_id, referred_id, status, created_at) VALUES (?, ?, 'pending', ?)
                    ON CONFLICT (referred_id) DO NOTHING''', (referrer['id'], user_id, now))
    return True


def reward_referral(conn, user_id, now):
    """On a referred account's first purchase, give both sides credits. Returns the referrer's reward, if any."""
    referral = conn.execute("SELECT * FROM referrals WHERE referred_id = ? AND status = 'pending'", (user_id,)).fetchone()
    if referral is None:
        return None
    if conn.execute("UPDATE referrals SET status = 'rewarded', rewarded_at = ? WHERE id = ? AND status = 'pending'",
                    (now, referral['id'])).rowcount != 1:
        return None
    add_credits(conn, user_id, REFERRED_CREDITS, now)
    rewarded = conn.execute("SELECT COUNT(*) AS n FROM referrals WHERE referrer_id = ? AND referrer_credits > 0",
                            (referral['referrer_id'],)).fetchone()['n']
    if rewarded >= REFERRAL_REWARD_LIMIT:
        conn.execute('UPDATE referrals SET referred_credits = ? WHERE id = ?', (REFERRED_CREDITS, referral['id']))
        return None
    add_credits(conn, referral['referrer_id'], REFERRER_CREDITS, now)
    conn.execute('UPDATE referrals SET referrer_credits = ?, referred_credits = ? WHERE id = ?',
                 (REFERRER_CREDITS, REFERRED_CREDITS, referral['id']))
    return {'referrer_id': referral['referrer_id'], 'credits': REFERRER_CREDITS}


def referral_summary(conn, user_id):
    stats = conn.execute('''SELECT COUNT(*) AS joined,
                                   SUM(CASE WHEN status = 'rewarded' THEN 1 ELSE 0 END) AS rewarded,
                                   SUM(referrer_credits) AS credits
                            FROM referrals WHERE referrer_id = ?''', (user_id,)).fetchone()
    return {'code': referral_code(conn, user_id), 'joined': stats['joined'] or 0, 'rewarded': stats['rewarded'] or 0,
            'credits_earned': stats['credits'] or 0, 'referrer_credits': REFERRER_CREDITS, 'referred_credits': REFERRED_CREDITS}

# ==========================================
#   REFUNDS (ADMIN)
# ==========================================

def refund_payment(payment_id, now, amount=None, revoke=True):
    """Refund a payment through Razorpay (all of what's left, or `amount`). A full refund can also take back
    what was bought. Returns the updated payment row."""
    conn = db.connect()
    try:
        payment = conn.execute('SELECT * FROM payments WHERE id = ?', (payment_id,)).fetchone()
    finally:
        conn.close()
    if payment is None or payment['status'] not in ('paid', 'refunded'):
        raise PaymentError('Payment not found.')
    if not payment['razorpay_payment_id']:
        raise PaymentError('This purchase was free, so there is nothing to refund.')
    refundable = payment['amount'] - (payment['refunded_amount'] or 0)
    amount = refundable if amount is None else amount
    if not 0 < amount <= refundable:
        raise PaymentError(f'You can refund up to {format_price(refundable, payment["currency"])}.')

    refund = razorpay_request('POST', f"/payments/{payment['razorpay_payment_id']}/refund",
                              {'amount': amount, 'notes': {'payment': str(payment_id)}})
    full = amount == refundable
    conn = db.connect()
    try:
        conn.execute('''UPDATE payments SET refunded_amount = refunded_amount + ?, refund_id = ?, status = ? WHERE id = ?''',
                     (amount, refund.get('id'), 'refunded' if full else payment['status'], payment_id))
        if full and revoke:
            revoke_product(conn, payment['user_id'], payment['plan_id'], now)
        conn.commit()
        return conn.execute('SELECT * FROM payments WHERE id = ?', (payment_id,)).fetchone()
    finally:
        conn.close()

# ==========================================
#   DASHBOARD
# ==========================================

def account_status(user_id, now):
    """Plan, credits, auto-renewal, recent payments and invoices for the dashboard."""
    conn = db.connect()
    try:
        sub = conn.execute('SELECT * FROM api_subscriptions WHERE user_id = ?', (user_id,)).fetchone()
        payments = conn.execute('''SELECT p.id, p.plan_id, p.currency, p.amount, p.status, p.paid_at, p.discount, p.coupon_code,
                                          p.refunded_amount, p.razorpay_subscription_id, i.id AS invoice_id, i.number AS invoice_number
                                   FROM payments p LEFT JOIN invoices i ON i.payment_id = p.id
                                   WHERE p.user_id = ? AND p.status IN ('paid', 'refunded')
                                   ORDER BY p.paid_at DESC LIMIT 20''', (user_id,)).fetchall()
        autopay = conn.execute('''SELECT * FROM autopay_subscriptions WHERE user_id = ? AND status IN ('authenticated', 'active', 'pending', 'halted')
                                  ORDER BY created_at DESC LIMIT 1''', (user_id,)).fetchone()
    finally:
        conn.close()

    plan = None
    credits = sub['credits'] if sub else 0
    if sub and sub['plan_id'] != NO_PLAN:
        current = db.to_datetime(sub['period_end']) > now
        plan = {
            'id': sub['plan_id'], 'name': plan_label(sub['plan_id']), 'cycle': sub['cycle'] or 'monthly',
            'requests_limit': sub['requests_limit'], 'requests_used': sub['requests_used'],
            'requests_remaining': max(0, sub['requests_limit'] - sub['requests_used']),
            'period_end': None if sub['plan_id'] == TRIAL_PLAN else db.timestamp_str(sub['period_end']),
            'active': current and sub['requests_used'] < sub['requests_limit'],
        }
    elif sub is None and trial_requests() > 0:
        plan = {'id': TRIAL_PLAN, 'name': 'Free trial', 'cycle': 'monthly', 'requests_limit': trial_requests(), 'requests_used': 0,
                'requests_remaining': trial_requests(), 'period_end': None, 'active': True}
    return {
        'plan': plan,
        'credits': credits,
        'can_use_api': bool(plan and plan['active']) or credits > 0,
        'autopay': {'product': autopay['product_id'], 'name': product_name(autopay['product_id']), 'status': autopay['status'],
                    'price': format_price(autopay['amount'], autopay['currency'])} if autopay else None,
        'autopay_available': autopay_enabled(),
        'payments': [{'id': p['id'], 'plan': product_name(p['plan_id']), 'product': p['plan_id'],
                      'amount': format_price(p['amount'], p['currency']),
                      'refunded': p['status'] == 'refunded', 'auto_renewal': bool(p['razorpay_subscription_id']),
                      'invoice_id': p['invoice_id'], 'invoice_number': p['invoice_number'],
                      'paid_at': db.timestamp_str(p['paid_at'])} for p in payments],
    }
