"""
Admin panel: look up users, see payments, issue refunds, grant plans and credits, manage coupons.

Admins are the accounts whose email is listed in ADMIN_EMAILS (comma-separated) and has been
verified (by the emailed link, Google/GitHub sign-in or a password reset), so nobody can claim
the role by signing up with an admin's address. Everyone else gets a 404, which doesn't reveal
that the panel exists. Every change is recorded in admin_actions.
"""
import json
import logging
import os
import re
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, abort, g, jsonify, render_template, request, session

import billing
import db

logger = logging.getLogger(__name__)

bp = Blueprint('admin', __name__, url_prefix='/admin')

COUPON_CODE = re.compile(r'^[A-Z0-9_-]{3,32}$')


def admin_emails():
    return {e.strip().lower() for e in os.environ.get('ADMIN_EMAILS', '').split(',') if e.strip()}


def current_admin():
    """The signed-in user if they're an admin, otherwise None."""
    user_id, emails = session.get('user_id'), admin_emails()
    if not user_id or not emails:
        return None
    conn = db.connect()
    try:
        user = conn.execute('SELECT id, username, email, email_verified FROM users WHERE id = ?', (user_id,)).fetchone()
    finally:
        conn.close()
    return user if user and user['email_verified'] and user['email'].lower() in emails else None


def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        g.admin = current_admin()
        if g.admin is None:
            abort(404)
        return f(*args, **kwargs)
    return wrapped


def body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def log_action(conn, action, target_user_id=None, **details):
    conn.execute('INSERT INTO admin_actions (admin_id, action, target_user_id, details, created_at) VALUES (?, ?, ?, ?, ?)',
                 (g.admin['id'], action, target_user_id, json.dumps(details), db.utcnow()))
    logger.info('Admin %s: %s (user %s)', g.admin['id'], action, target_user_id)


def ts(value):
    return db.timestamp_str(value)


def to_minor(value):
    """'99' or '99.50' (rupees / dollars) -> 9900 / 9950, or None if not a positive amount."""
    try:
        amount = round(float(str(value).strip()) * 100)
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None


def payment_dict(p):
    return {'id': p['id'], 'user_id': p['user_id'], 'email': p['email'], 'product': billing.product_name(p['plan_id']),
            'amount': billing.format_price(p['amount'], p['currency']), 'currency': p['currency'],
            'refunded': billing.format_price(p['refunded_amount'] or 0, p['currency']) if p['refunded_amount'] else None,
            'refundable': (p['amount'] - (p['refunded_amount'] or 0)) / 100 if p['razorpay_payment_id'] else 0,
            'status': p['status'], 'coupon': p['coupon_code'], 'auto_renewal': bool(p['razorpay_subscription_id']),
            'razorpay_payment_id': p['razorpay_payment_id'], 'invoice_id': p['invoice_id'], 'invoice_number': p['invoice_number'],
            'created_at': ts(p['created_at']), 'paid_at': ts(p['paid_at'])}


PAYMENT_QUERY = '''SELECT p.*, u.email, i.id AS invoice_id, i.number AS invoice_number
                   FROM payments p JOIN users u ON u.id = p.user_id LEFT JOIN invoices i ON i.payment_id = p.id'''

# ==========================================
#   PAGES
# ==========================================

@bp.route('')
@admin_required
def panel():
    products = [{'id': pid, 'name': p['name']} for pid, p in billing.PRODUCTS.items()]
    return render_template('admin.html', products=products)


@bp.route('/api/overview')
@admin_required
def overview():
    now = db.utcnow()
    conn = db.connect()
    try:
        users = conn.execute('''SELECT COUNT(*) AS total, SUM(CASE WHEN email_verified = 1 THEN 1 ELSE 0 END) AS verified,
                                       SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS week
                                FROM users''', (now - timedelta(days=7),)).fetchone()
        paying = conn.execute(f'''SELECT COUNT(*) AS n FROM api_subscriptions
                                  WHERE plan_id IN ({",".join("?" * len(billing.PLANS))}) AND period_end > ?''',
                              (*billing.PLANS, now)).fetchone()['n']
        autopay = conn.execute(f'''SELECT COUNT(*) AS n FROM autopay_subscriptions
                                   WHERE status IN ({",".join("?" * len(billing.AUTOPAY_LIVE))})''', billing.AUTOPAY_LIVE).fetchone()['n']
        revenue = conn.execute('''SELECT currency, SUM(amount - refunded_amount) AS total, COUNT(*) AS n FROM payments
                                  WHERE status IN ('paid', 'refunded') AND paid_at >= ? GROUP BY currency''',
                               (now - timedelta(days=30),)).fetchall()
        requests_day = conn.execute('SELECT COUNT(*) AS n FROM api_usage WHERE timestamp >= ?', (now - timedelta(days=1),)).fetchone()['n']
    finally:
        conn.close()
    return jsonify({
        'users': users['total'] or 0, 'verified': users['verified'] or 0, 'signups_week': users['week'] or 0,
        'paying': paying, 'autopay': autopay, 'requests_day': requests_day,
        'revenue_30d': [{'currency': r['currency'], 'total': billing.format_price(r['total'] or 0, r['currency']), 'payments': r['n']}
                        for r in revenue],
    })

# ==========================================
#   USERS
# ==========================================

@bp.route('/api/users')
@admin_required
def search_users():
    q = (request.args.get('q') or '').strip().lower()[:100]
    conn = db.connect()
    try:
        sql = '''SELECT u.id, u.username, u.email, u.created_at, u.email_verified, s.plan_id, s.period_end, s.credits
                 FROM users u LEFT JOIN api_subscriptions s ON s.user_id = u.id'''
        if q:
            rows = conn.execute(sql + ' WHERE lower(u.email) LIKE ? OR lower(u.username) LIKE ? OR CAST(u.id AS TEXT) = ? '
                                      'ORDER BY u.id DESC LIMIT 50', (f'%{q}%', f'%{q}%', q)).fetchall()
        else:
            rows = conn.execute(sql + ' ORDER BY u.id DESC LIMIT 50').fetchall()
    finally:
        conn.close()
    return jsonify({'users': [{'id': r['id'], 'username': r['username'], 'email': r['email'], 'created_at': ts(r['created_at']),
                               'verified': bool(r['email_verified']), 'plan': billing.plan_label(r['plan_id']) if r['plan_id'] else None,
                               'credits': r['credits'] or 0} for r in rows]})


@bp.route('/api/users/<int:user_id>')
@admin_required
def user_detail(user_id):
    now = db.utcnow()
    conn = db.connect()
    try:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if user is None:
            return jsonify({'error': 'User not found.'}), 404
        providers = [r['provider'] for r in conn.execute('SELECT provider FROM oauth_accounts WHERE user_id = ?', (user_id,))]
        sub = conn.execute('SELECT * FROM api_subscriptions WHERE user_id = ?', (user_id,)).fetchone()
        payments = conn.execute(PAYMENT_QUERY + ' WHERE p.user_id = ? ORDER BY p.created_at DESC LIMIT 50', (user_id,)).fetchall()
        autopay = conn.execute('SELECT * FROM autopay_subscriptions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5', (user_id,)).fetchall()
        keys = conn.execute('SELECT COUNT(*) AS n FROM api_keys WHERE user_id = ? AND is_active = 1', (user_id,)).fetchone()['n']
        usage = conn.execute('''SELECT COUNT(*) AS total, SUM(CASE WHEN u.timestamp >= ? THEN 1 ELSE 0 END) AS month
                                FROM api_usage u JOIN api_keys k ON k.id = u.api_key_id WHERE k.user_id = ?''',
                             (now - timedelta(days=30), user_id)).fetchone()
        referrals = conn.execute('''SELECT r.status, r.created_at, u.email FROM referrals r JOIN users u ON u.id = r.referred_id
                                    WHERE r.referrer_id = ? ORDER BY r.created_at DESC LIMIT 20''', (user_id,)).fetchall()
        referrer = conn.execute('SELECT id, email FROM users WHERE id = ?', (user['referred_by'],)).fetchone() if user['referred_by'] else None
        profile = conn.execute('SELECT * FROM billing_profiles WHERE user_id = ?', (user_id,)).fetchone()
    finally:
        conn.close()

    return jsonify({
        'user': {'id': user['id'], 'username': user['username'], 'email': user['email'], 'created_at': ts(user['created_at']),
                 'verified': bool(user['email_verified']), 'providers': providers,
                 'password_login': user['password'] != '!oauth', 'referral_code': user['referral_code'],
                 'referred_by': {'id': referrer['id'], 'email': referrer['email']} if referrer else None},
        'subscription': {'plan': billing.plan_label(sub['plan_id']), 'cycle': sub['cycle'], 'requests_limit': sub['requests_limit'],
                         'requests_used': sub['requests_used'], 'credits': sub['credits'], 'period_end': ts(sub['period_end']),
                         'active': db.to_datetime(sub['period_end']) > now and sub['requests_used'] < sub['requests_limit']} if sub else None,
        'autopay': [{'product': billing.product_name(a['product_id']), 'status': a['status'], 'id': a['razorpay_subscription_id'],
                     'created_at': ts(a['created_at'])} for a in autopay],
        'payments': [payment_dict(p) for p in payments],
        'keys': keys, 'requests_total': usage['total'] or 0, 'requests_30d': usage['month'] or 0,
        'referrals': [{'email': r['email'], 'status': r['status'], 'created_at': ts(r['created_at'])} for r in referrals],
        'billing_profile': {'business_name': profile['business_name'], 'gstin': profile['gstin'], 'state_code': profile['state_code'],
                            'address': profile['address']} if profile else None,
    })


@bp.route('/api/users/<int:user_id>/grant', methods=['POST'])
@admin_required
def grant(user_id):
    data = body()
    product_id = data.get('product')
    note = str(data.get('note') or '')[:200]
    if product_id not in billing.PRODUCTS:
        return jsonify({'error': 'Choose a plan or top-up.'}), 400
    conn = db.connect()
    try:
        if not conn.execute('SELECT 1 FROM users WHERE id = ?', (user_id,)).fetchone():
            return jsonify({'error': 'User not found.'}), 404
        billing.grant_product(conn, user_id, product_id, db.utcnow())
        log_action(conn, 'grant', user_id, product=product_id, note=note)
        conn.commit()
    finally:
        conn.close()
    return jsonify({'success': True, 'message': f'Granted {billing.product_name(product_id)}.'})


@bp.route('/api/users/<int:user_id>/credits', methods=['POST'])
@admin_required
def add_credits(user_id):
    data = body()
    try:
        credits = int(data.get('credits'))
    except (TypeError, ValueError):
        credits = 0
    note = str(data.get('note') or '')[:200]
    if not 0 < credits <= 1_000_000:
        return jsonify({'error': 'Enter a number of requests between 1 and 1,000,000.'}), 400
    conn = db.connect()
    try:
        if not conn.execute('SELECT 1 FROM users WHERE id = ?', (user_id,)).fetchone():
            return jsonify({'error': 'User not found.'}), 404
        billing.add_credits(conn, user_id, credits, db.utcnow())
        log_action(conn, 'credits', user_id, credits=credits, note=note)
        conn.commit()
    finally:
        conn.close()
    return jsonify({'success': True, 'message': f'Added {credits:,} requests.'})


@bp.route('/api/users/<int:user_id>/verify', methods=['POST'])
@admin_required
def verify_user(user_id):
    conn = db.connect()
    try:
        updated = conn.execute('UPDATE users SET email_verified = 1 WHERE id = ?', (user_id,)).rowcount
        if updated:
            log_action(conn, 'verify_email', user_id)
        conn.commit()
    finally:
        conn.close()
    if not updated:
        return jsonify({'error': 'User not found.'}), 404
    return jsonify({'success': True, 'message': 'Email marked as verified.'})

# ==========================================
#   PAYMENTS
# ==========================================

@bp.route('/api/payments')
@admin_required
def list_payments():
    status = request.args.get('status') or 'paid,refunded'
    statuses = [s for s in status.split(',') if s in ('created', 'paid', 'refunded')] or ['paid', 'refunded']
    q = (request.args.get('q') or '').strip().lower()[:100]
    params = list(statuses)
    sql = PAYMENT_QUERY + f' WHERE p.status IN ({",".join("?" * len(statuses))})'
    if q:
        sql += ' AND (lower(u.email) LIKE ? OR p.razorpay_payment_id = ? OR p.razorpay_order_id = ? OR lower(i.number) = ?)'
        params += [f'%{q}%', q, q, q]
    conn = db.connect()
    try:
        rows = conn.execute(sql + ' ORDER BY p.created_at DESC LIMIT 100', params).fetchall()
    finally:
        conn.close()
    return jsonify({'payments': [payment_dict(p) for p in rows]})


@bp.route('/api/payments/<int:payment_id>/refund', methods=['POST'])
@admin_required
def refund(payment_id):
    data = body()
    amount = None
    if str(data.get('amount') or '').strip():
        amount = to_minor(data.get('amount'))
        if amount is None:
            return jsonify({'error': 'Enter a valid amount, or leave it empty for a full refund.'}), 400
    revoke = data.get('revoke', True) is not False
    try:
        payment = billing.refund_payment(payment_id, db.utcnow(), amount, revoke)
    except billing.PaymentError as e:
        return jsonify({'error': str(e)}), 400
    conn = db.connect()
    try:
        log_action(conn, 'refund', payment['user_id'], payment=payment_id, amount=amount or 'full', revoke=revoke,
                   refund_id=payment['refund_id'])
        conn.commit()
    finally:
        conn.close()
    return jsonify({'success': True, 'message': 'Refund issued. Razorpay returns the money in 5-7 working days.'})

# ==========================================
#   COUPONS
# ==========================================

@bp.route('/api/coupons')
@admin_required
def list_coupons():
    conn = db.connect()
    try:
        rows = conn.execute('SELECT * FROM coupons ORDER BY created_at DESC LIMIT 200').fetchall()
    finally:
        conn.close()
    return jsonify({'coupons': [{'code': c['code'], 'percent_off': c['percent_off'], 'max_redemptions': c['max_redemptions'],
                                 'redeemed': c['redeemed_count'], 'expires_at': ts(c['expires_at']), 'applies_to': c['applies_to'] or '',
                                 'active': bool(c['active'])} for c in rows]})


@bp.route('/api/coupons', methods=['POST'])
@admin_required
def create_coupon():
    data = body()
    code = billing.normalize_code(data.get('code'))
    if not COUPON_CODE.match(code):
        return jsonify({'error': 'Use 3-32 letters, digits, "-" or "_" for the code.'}), 400
    try:
        percent = int(data.get('percent_off'))
    except (TypeError, ValueError):
        percent = 0
    if not 1 <= percent <= 100:
        return jsonify({'error': 'The discount must be between 1% and 100%.'}), 400
    max_redemptions = None
    if str(data.get('max_redemptions') or '').strip():
        try:
            max_redemptions = int(data['max_redemptions'])
        except (TypeError, ValueError):
            max_redemptions = 0
        if max_redemptions < 1:
            return jsonify({'error': 'Maximum uses must be a positive number, or empty for unlimited.'}), 400
    expires_at = None
    if str(data.get('expires_at') or '').strip():
        try:
            # The coupon works through the end of that day (UTC)
            expires_at = datetime.strptime(str(data['expires_at']).strip(), '%Y-%m-%d') + timedelta(days=1)
        except ValueError:
            return jsonify({'error': 'Use YYYY-MM-DD for the expiry date.'}), 400
    targets = [t.strip() for t in str(data.get('applies_to') or '').split(',') if t.strip()]
    allowed = set(billing.PRODUCTS) | {'plans', 'monthly', 'yearly'}
    if any(t not in allowed for t in targets):
        return jsonify({'error': 'Unknown product in "applies to".'}), 400

    conn = db.connect()
    try:
        conn.execute('''INSERT INTO coupons (code, percent_off, max_redemptions, redeemed_count, expires_at, active, applies_to, created_at)
                        VALUES (?, ?, ?, 0, ?, 1, ?, ?)''',
                     (code, percent, max_redemptions, expires_at, ','.join(targets) or None, db.utcnow()))
        log_action(conn, 'coupon_create', None, code=code, percent_off=percent)
        conn.commit()
    except db.IntegrityError:
        return jsonify({'error': 'A coupon with that code already exists.'}), 400
    finally:
        conn.close()
    return jsonify({'success': True, 'message': f'Coupon {code} created.'})


@bp.route('/api/coupons/<code>/toggle', methods=['POST'])
@admin_required
def toggle_coupon(code):
    code = billing.normalize_code(code)
    conn = db.connect()
    try:
        updated = conn.execute('UPDATE coupons SET active = 1 - active WHERE code = ?', (code,)).rowcount
        if updated:
            log_action(conn, 'coupon_toggle', None, code=code)
        conn.commit()
    finally:
        conn.close()
    if not updated:
        return jsonify({'error': 'Coupon not found.'}), 404
    return jsonify({'success': True})

# ==========================================
#   AUDIT LOG
# ==========================================

@bp.route('/api/actions')
@admin_required
def actions():
    conn = db.connect()
    try:
        rows = conn.execute('''SELECT a.*, u.email AS admin_email, t.email AS target_email FROM admin_actions a
                               LEFT JOIN users u ON u.id = a.admin_id LEFT JOIN users t ON t.id = a.target_user_id
                               ORDER BY a.id DESC LIMIT 100''').fetchall()
    finally:
        conn.close()
    return jsonify({'actions': [{'action': r['action'], 'admin': r['admin_email'], 'target': r['target_email'],
                                 'target_user_id': r['target_user_id'], 'details': json.loads(r['details'] or '{}'),
                                 'created_at': ts(r['created_at'])} for r in rows]})
