"""
GST invoices (PDF) for API purchases.

Every paid purchase gets one invoice, numbered in an unbroken series per Indian financial
year (April to March, in IST), e.g. INV/2627/00001. The number stays within GST's
16-character limit.

  - BUSINESS_GSTIN set, payment in rupees: a tax invoice. Prices include 18% GST, split into
    CGST + SGST when the customer is in the seller's state (or their state isn't known, in
    which case the place of supply is the seller's location) and IGST otherwise.
  - BUSINESS_GSTIN set, payment in US dollars: an export of services, zero-rated under a
    Letter of Undertaking (set GST_LUT_ARN to print its reference).
  - No BUSINESS_GSTIN: a plain receipt without tax lines.

Customers who want their GSTIN on the invoice add it under Dashboard -> Billing details
before paying; invoices are issued at the time of payment and aren't changed afterwards.
Have an accountant confirm the SAC code (GST_SAC_CODE) and the tax treatment before going live.

PDFs are drawn on demand with fpdf2, imported lazily (it adds ~40 MB). The built-in PDF fonts
cover Latin-1 only, so other characters in names and addresses print as "?".
"""
import logging
import os
import re
import unicodedata
from datetime import timedelta, timezone

import db

logger = logging.getLogger(__name__)

GST_RATE = 18
IST = timezone(timedelta(hours=5, minutes=30))

GSTIN_PATTERN = re.compile(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$')
GSTIN_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# GST state codes (first two digits of a GSTIN)
STATES = {
    '01': 'Jammu and Kashmir', '02': 'Himachal Pradesh', '03': 'Punjab', '04': 'Chandigarh', '05': 'Uttarakhand',
    '06': 'Haryana', '07': 'Delhi', '08': 'Rajasthan', '09': 'Uttar Pradesh', '10': 'Bihar', '11': 'Sikkim',
    '12': 'Arunachal Pradesh', '13': 'Nagaland', '14': 'Manipur', '15': 'Mizoram', '16': 'Tripura', '17': 'Meghalaya',
    '18': 'Assam', '19': 'West Bengal', '20': 'Jharkhand', '21': 'Odisha', '22': 'Chhattisgarh', '23': 'Madhya Pradesh',
    '24': 'Gujarat', '26': 'Dadra and Nagar Haveli and Daman and Diu', '27': 'Maharashtra', '29': 'Karnataka', '30': 'Goa',
    '31': 'Lakshadweep', '32': 'Kerala', '33': 'Tamil Nadu', '34': 'Puducherry', '35': 'Andaman and Nicobar Islands',
    '36': 'Telangana', '37': 'Andhra Pradesh', '38': 'Ladakh', '97': 'Other Territory',
}


def config(name, default=''):
    return os.environ.get(name, default)


def seller_gstin():
    return config('BUSINESS_GSTIN').strip().upper()


def number_prefix():
    # Letters and digits only, at most 5, so "PREFIX/2627/00001" fits in 16 characters
    return re.sub(r'[^A-Z0-9]', '', config('INVOICE_PREFIX', 'INV').upper())[:5] or 'INV'


def gstin_is_valid(gstin):
    """Format and check digit (base-36 Luhn variant used by the GST network)."""
    if not GSTIN_PATTERN.match(gstin or '') or gstin[:2] not in STATES:
        return False
    total = 0
    for i, char in enumerate(gstin[:14]):
        product = GSTIN_CHARS.index(char) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return GSTIN_CHARS[(36 - total % 36) % 36] == gstin[14]


def financial_year(moment):
    """'2627' for April 2026 to March 2027, from a naive UTC timestamp, in Indian time."""
    local = moment.replace(tzinfo=timezone.utc).astimezone(IST)
    start = local.year if local.month >= 4 else local.year - 1
    return f'{start % 100:02d}{(start + 1) % 100:02d}'


def tax_breakdown(total, currency, buyer_state, seller_state):
    """Split a GST-inclusive total (smallest currency unit) into taxable value and taxes."""
    taxes = {'kind': 'receipt', 'taxable': total, 'cgst': 0, 'sgst': 0, 'igst': 0}
    if not seller_state:
        return taxes
    if currency != 'INR':
        return {**taxes, 'kind': 'export'}
    if buyer_state and buyer_state != seller_state:
        igst = round(total * GST_RATE / (100 + GST_RATE))
        return {**taxes, 'kind': 'tax', 'taxable': total - igst, 'igst': igst}
    half = round(total * GST_RATE / 2 / (100 + GST_RATE))
    return {**taxes, 'kind': 'tax', 'taxable': total - 2 * half, 'cgst': half, 'sgst': half}

# ==========================================
#   BILLING DETAILS
# ==========================================

def get_profile(conn, user_id):
    row = conn.execute('SELECT * FROM billing_profiles WHERE user_id = ?', (user_id,)).fetchone()
    return {'business_name': row['business_name'] or '', 'gstin': row['gstin'] or '', 'state_code': row['state_code'] or '',
            'address': row['address'] or ''} if row else {'business_name': '', 'gstin': '', 'state_code': '', 'address': ''}


def save_profile(conn, user_id, business_name, gstin, state_code, address, now):
    """Validate and store billing details. Returns an error message, or None."""
    gstin = re.sub(r'\s+', '', gstin or '').upper()
    if gstin:
        if not gstin_is_valid(gstin):
            return "That GSTIN doesn't look right. Check it and try again."
        if not business_name:
            return 'Enter the business name registered with this GSTIN.'
        state_code = gstin[:2]
    if state_code and state_code not in STATES:
        return 'Choose a valid state.'
    if len(business_name) > 120 or len(address) > 300:
        return 'Business name or address is too long.'
    conn.execute('''INSERT INTO billing_profiles (user_id, business_name, gstin, state_code, address, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (user_id) DO UPDATE SET business_name = excluded.business_name, gstin = excluded.gstin,
                        state_code = excluded.state_code, address = excluded.address, updated_at = excluded.updated_at''',
                 (user_id, business_name or None, gstin or None, state_code or None, address or None, now))
    return None

# ==========================================
#   ISSUING
# ==========================================

def create_invoice(payment_id, description):
    """Issue the invoice for a paid purchase, once. Returns the invoice row (or None if not paid)."""
    conn = db.connect()
    try:
        existing = conn.execute('SELECT * FROM invoices WHERE payment_id = ?', (payment_id,)).fetchone()
        if existing:
            return existing
        payment = conn.execute('''SELECT p.*, u.username, u.email FROM payments p JOIN users u ON u.id = p.user_id
                                  WHERE p.id = ? AND p.status = 'paid' ''', (payment_id,)).fetchone()
        if payment is None:
            return None
        profile = get_profile(conn, payment['user_id'])
        gstin = seller_gstin()
        seller_state = gstin[:2] if gstin else None
        taxes = tax_breakdown(payment['amount'], payment['currency'], profile['state_code'], seller_state)
        issued_at = db.to_datetime(payment['paid_at'])
        fy = financial_year(issued_at)
        place = 'Outside India' if taxes['kind'] == 'export' else (profile['state_code'] or seller_state)

        for _ in range(3):
            try:
                # The year's counter row is locked until commit, so parallel invoices queue up instead of
                # colliding, and a failed insert rolls the counter back: numbers stay unique and gapless
                seq = conn.execute('''INSERT INTO invoice_counters (fy, last_seq)
                                      SELECT ?, COALESCE(MAX(seq), 0) + 1 FROM invoices WHERE fy = ?
                                      ON CONFLICT (fy) DO UPDATE SET last_seq = invoice_counters.last_seq + 1
                                      RETURNING last_seq''', (fy, fy)).fetchone()['last_seq']
                conn.execute('''INSERT INTO invoices (payment_id, user_id, number, fy, seq, kind, issued_at, currency, total,
                                    taxable, cgst, sgst, igst, place_of_supply, sac, description,
                                    seller_name, seller_address, seller_gstin,
                                    buyer_name, buyer_email, buyer_business, buyer_gstin, buyer_state, buyer_address)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                             (payment_id, payment['user_id'], f'{number_prefix()}/{fy}/{seq:05d}', fy, seq, taxes['kind'],
                              issued_at, payment['currency'], payment['amount'], taxes['taxable'], taxes['cgst'], taxes['sgst'],
                              taxes['igst'], place, config('GST_SAC_CODE') or '998314', description,
                              config('BUSINESS_NAME', 'AI Humanizer'), config('BUSINESS_ADDRESS'), gstin or None,
                              payment['username'], payment['email'], profile['business_name'] or None,
                              profile['gstin'] or None, profile['state_code'] or None, profile['address'] or None))
                conn.commit()
                return conn.execute('SELECT * FROM invoices WHERE payment_id = ?', (payment_id,)).fetchone()
            except db.IntegrityError:
                # This payment was invoiced by another request at the same moment
                conn.rollback()
                existing = conn.execute('SELECT * FROM invoices WHERE payment_id = ?', (payment_id,)).fetchone()
                if existing:
                    return existing
        logger.error('Could not number the invoice for payment %s', payment_id)
        return None
    finally:
        conn.close()

# ==========================================
#   PDF
# ==========================================

ONES = ('Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen '
        'Seventeen Eighteen Nineteen').split()
TENS = 'Zero Ten Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()


def _words(n):
    """Indian numbering: 1,23,45,678 -> One Crore Twenty-Three Lakh ..."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ('-' + ONES[n % 10] if n % 10 else '')
    for size, name in ((10_000_000, 'Crore'), (100_000, 'Lakh'), (1000, 'Thousand'), (100, 'Hundred')):
        if n >= size:
            head, rest = divmod(n, size)
            return f'{_words(head)} {name}' + (f' {_words(rest)}' if rest else '')


def amount_in_words(amount, currency):
    major, minor = divmod(amount, 100)
    unit, sub = ('Rupees', 'Paise') if currency == 'INR' else ('US Dollars', 'Cents')
    text = f'{unit} {_words(major)}'
    if minor:
        text += f' and {_words(minor)} {sub}'
    return text + ' Only'


def money(amount, currency):
    return f'{currency} {amount / 100:,.2f}'


def latin1(text):
    """The PDF's built-in fonts are Latin-1 only."""
    text = unicodedata.normalize('NFKC', str(text or ''))
    return text.encode('latin-1', 'replace').decode('latin-1')


def state_label(code):
    return f'{STATES[code]} ({code})' if code in STATES else (code or '')


def render_pdf(invoice, refunded_amount=0):
    """The invoice as PDF bytes."""
    from fpdf import FPDF

    ink, muted, accent, line = (22, 23, 27), (110, 113, 122), (240, 84, 44), (225, 221, 212)
    currency = invoice['currency']
    title = {'tax': 'TAX INVOICE', 'export': 'INVOICE', 'receipt': 'RECEIPT'}[invoice['kind']]

    pdf = FPDF(format='A4')
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 18)
    pdf.set_title(f"{title.title()} {invoice['number']}")
    pdf.set_author(latin1(invoice['seller_name']))
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def text(size, value, style='', color=ink, w=0, h=None, align='L', ln=True):
        pdf.set_font('Helvetica', style, size)
        pdf.set_text_color(*color)
        pdf.cell(w, h or size * 0.5, latin1(value), align=align, new_x='LMARGIN' if ln else 'RIGHT', new_y='NEXT' if ln else 'TOP')

    def rule(gap=4):
        pdf.ln(gap)
        pdf.set_draw_color(*line)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(gap)

    # Seller and document title
    top = pdf.get_y()
    pdf.set_fill_color(*accent)
    pdf.rect(pdf.l_margin, top, 3, 12, style='F')
    pdf.set_x(pdf.l_margin + 6)
    text(16, invoice['seller_name'], 'B', w=width * 0.6 - 6, h=7, ln=False)
    text(16, title, 'B', accent, w=width * 0.4, h=7, align='R')
    pdf.set_x(pdf.l_margin + 6)
    for detail in filter(None, [invoice['seller_address'],
                                f"GSTIN: {invoice['seller_gstin']} · State: {state_label(invoice['seller_gstin'][:2])}"
                                if invoice['seller_gstin'] else None,
                                config('CONTACT_EMAIL')]):
        pdf.set_x(pdf.l_margin + 6)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*muted)
        pdf.multi_cell(width * 0.6 - 6, 4.5, latin1(detail), align='L', new_x='LMARGIN', new_y='NEXT')
    pdf.set_y(max(pdf.get_y(), top + 14))
    rule()

    # Customer and invoice details side by side
    column = width / 2
    y = pdf.get_y()
    text(8, 'BILLED TO', 'B', muted)
    pdf.ln(1)
    billed = [invoice['buyer_business'], invoice['buyer_name'] if not invoice['buyer_business'] else f"Attn: {invoice['buyer_name']}",
              invoice['buyer_address'], f"GSTIN: {invoice['buyer_gstin']}" if invoice['buyer_gstin'] else None,
              f"State: {state_label(invoice['buyer_state'])}" if invoice['buyer_state'] else None, invoice['buyer_email']]
    for i, detail in enumerate(filter(None, billed)):
        pdf.set_font('Helvetica', 'B' if i == 0 else '', 10)
        pdf.set_text_color(*ink)
        pdf.multi_cell(column - 6, 5, latin1(detail), align='L', new_x='LMARGIN', new_y='NEXT')
    left_end = pdf.get_y()

    pdf.set_y(y)
    place = invoice['place_of_supply']
    facts = [('Invoice no.', invoice['number']),
             ('Date', db.to_datetime(invoice['issued_at']).replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y')),
             ('Payment ID', invoice['payment_ref'] or '-')]
    if invoice['kind'] != 'receipt':
        facts.append(('Place of supply', place if place == 'Outside India' else state_label(place)))
    for label, value in facts:
        pdf.set_x(pdf.l_margin + column)
        text(9, label, '', muted, w=32, h=6, ln=False)
        text(10, value, 'B', w=column - 32, h=6, align='R')
    pdf.set_y(max(left_end, pdf.get_y()))
    pdf.ln(6)

    # Line item
    widths = (width - 70, 22, 14, 34)
    pdf.set_fill_color(243, 241, 236)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*muted)
    for label, w, align in zip(('Description', 'SAC', 'Qty', f'Amount ({currency})'), widths, 'LCCR'):
        pdf.cell(w, 8, label, align=align, fill=True)
    pdf.ln(8)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*ink)
    for value, w, align in zip((invoice['description'], invoice['sac'] if invoice['kind'] != 'receipt' else '-', '1',
                                f"{invoice['taxable'] / 100:,.2f}"), widths, 'LCCR'):
        pdf.cell(w, 10, latin1(value), align=align)
    pdf.ln(10)
    rule(2)

    # Totals
    rows = []
    if invoice['kind'] == 'tax':
        rows.append(('Taxable value', invoice['taxable']))
        if invoice['igst']:
            rows.append((f'IGST @ {GST_RATE}%', invoice['igst']))
        else:
            rows += [(f'CGST @ {GST_RATE / 2:g}%', invoice['cgst']), (f'SGST @ {GST_RATE / 2:g}%', invoice['sgst'])]
    elif invoice['kind'] == 'export':
        rows += [('Taxable value', invoice['taxable']), ('IGST @ 0% (export under LUT)', 0)]
    for label, amount in rows:
        pdf.set_x(pdf.l_margin + width - 95)
        text(10, label, '', muted, w=60, h=6.5, ln=False)
        text(10, money(amount, currency), '', w=35, h=6.5, align='R')
    pdf.set_x(pdf.l_margin + width - 95)
    pdf.set_draw_color(*ink)
    pdf.line(pdf.get_x(), pdf.get_y() + 1, pdf.l_margin + width, pdf.get_y() + 1)
    pdf.ln(2)
    pdf.set_x(pdf.l_margin + width - 95)
    text(12, 'Total', 'B', w=60, h=9, ln=False)
    text(12, money(invoice['total'], currency), 'B', w=35, h=9, align='R')
    if refunded_amount:
        pdf.set_x(pdf.l_margin + width - 95)
        text(10, 'Refunded', '', muted, w=60, h=6.5, ln=False)
        text(10, '-' + money(refunded_amount, currency), '', muted, w=35, h=6.5, align='R')
    pdf.ln(4)
    text(9, 'Amount in words', 'B', muted, h=5)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*ink)
    pdf.multi_cell(width, 5, latin1(amount_in_words(invoice['total'], currency)), align='L', new_x='LMARGIN', new_y='NEXT')

    # Notes
    notes = []
    if invoice['kind'] == 'tax':
        notes += ['Prices include GST.', 'Tax payable on reverse charge basis: No.']
    elif invoice['kind'] == 'export':
        lut = config('GST_LUT_ARN')
        notes.append('Supply of services meant for export under Letter of Undertaking without payment of IGST.'
                     + (f' LUT ARN: {lut}.' if lut else ''))
    else:
        notes.append('This is a payment receipt, not a tax invoice.')
    notes.append('This is a computer-generated document and does not need a signature.')
    pdf.ln(6)
    rule(2)
    for note in notes:
        pdf.set_font('Helvetica', '', 8.5)
        pdf.set_text_color(*muted)
        pdf.multi_cell(width, 4.5, latin1(note), align='L', new_x='LMARGIN', new_y='NEXT')
    return bytes(pdf.output())


def pdf_for(invoice_id, user_id=None):
    """(filename, PDF bytes) for an invoice, or None. With user_id, only that user's invoice."""
    conn = db.connect()
    try:
        invoice = conn.execute('''SELECT i.*, p.razorpay_payment_id AS payment_ref, p.refunded_amount
                                  FROM invoices i JOIN payments p ON p.id = i.payment_id WHERE i.id = ?''', (invoice_id,)).fetchone()
    finally:
        conn.close()
    if invoice is None or (user_id is not None and invoice['user_id'] != user_id):
        return None
    filename = invoice['number'].replace('/', '-') + '.pdf'
    return filename, render_pdf(invoice, invoice['refunded_amount'] or 0)
