"""Regenerate the three sample PDFs (dev tool; the generated PDFs are committed).

Requires reportlab (``pip install reportlab``) - NOT needed to run the application.
One logical page of text per PDF page so that page numbers in citations are exact.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_documents"

DOCS = {
    "finance/expense_policy.pdf": ("Meridian Cloud Expense Policy", [
        ("1. Scope and principles",
         "This policy applies to all employees who spend company money on business purposes. Employees are expected "
         "to spend as if the money were their own. Expenses are reimbursed only when they are reasonable, necessary "
         "for business and supported by documentation. Personal expenses, fines and alcohol consumed without clients "
         "are never reimbursed. Spouse or family travel costs are not covered unless a vice president approves them in "
         "writing before the trip.\n"
         "Meals while travelling are reimbursed up to the following limits per person: breakfast up to $15, lunch up "
         "to $25 and dinner up to $60. When entertaining customers, the dinner limit rises to $90 per person and the "
         "most senior employee at the table pays and claims the expense. Meals on regular office days are not "
         "reimbursable."),
        ("2. Travel and accommodation",
         "Flights must be booked in economy class through the company travel portal at least 14 days before departure "
         "when possible. Business class is allowed only for flights longer than 8 hours and requires vice president "
         "approval before booking. For flights shorter than 6 hours the economy fare is mandatory with no exceptions.\n"
         "Hotel rooms are capped at $220 per night in tier-1 cities, which are London, New York, San Francisco and "
         "Tokyo. In all other cities the cap is $150 per night. Employees who exceed the cap must pay the difference "
         "themselves. Rental cars are permitted only when public transport is not practical, and the vehicle class "
         "must be compact or mid-size. Mileage for personal vehicles is reimbursed at $0.60 per mile."),
        ("3. Claims and approvals",
         "Expense reports must be submitted through ExpensePortal within 30 days of the expense date. Claims older "
         "than 30 days are rejected unless the finance team grants an exception. A receipt is required for every "
         "expense above $25. For expenses of $25 or less, a written description is sufficient.\n"
         "A direct manager can approve claims up to $2,000. Claims above $2,000 additionally need approval from the "
         "finance director. Approved claims are paid within 10 business days in the next payroll run. The corporate "
         "card must be used for travel bookings, and card statements are reconciled monthly. Employees who leave the "
         "company must submit all outstanding claims before their last working day."),
    ]),
    "security/password_and_access_policy.pdf": ("Meridian Cloud Password and Access Policy", [
        ("1. Passwords and authentication",
         "All employees must use a password of at least 14 characters. Passwords must not be reused across company "
         "systems and must be stored in the approved password manager. User passwords are not rotated on a fixed "
         "schedule; they must be changed immediately when a compromise is suspected. Service account secrets and API "
         "keys must be rotated every 90 days and are stored only in the secrets vault, never in source code.\n"
         "Multi-factor authentication is mandatory for every employee account. Administrators and anyone with "
         "production access must use a hardware security key; authenticator apps are not sufficient for these roles. "
         "Interactive sessions expire after 12 hours of inactivity."),
        ("2. Access control and logging",
         "Access follows the principle of least privilege. Managers review the access rights of their team every "
         "quarter, and unused accounts are disabled after 60 days of inactivity. Privileged production access is "
         "granted just in time for a maximum of 4 hours and every grant must reference a ticket.\n"
         "Authentication and access logs are collected centrally and retained for 400 days. Logs are write-once and "
         "can be read only by the Security team. Alerts fire on repeated failed logins, logins from impossible "
         "locations and use of break-glass accounts. When an employee leaves, all of their access must be revoked "
         "within 1 hour of their departure time."),
    ]),
    "support/sla_and_support_tiers.pdf": ("Atlas Support Tiers and Service Level Agreement", [
        ("1. Support tiers",
         "Atlas customers choose one of three support tiers. Basic support is delivered by email and the first "
         "response target is 1 business day. Business support is available 24 hours a day and guarantees a first "
         "response within 1 hour for critical issues. Enterprise support promises a first response within 15 minutes "
         "for critical issues and includes a dedicated technical account manager and a quarterly architecture review.\n"
         "An issue is critical when production is unavailable or data loss is suspected. Customers can open critical "
         "tickets by phone on the Business and Enterprise tiers. Issues are escalated to engineering if they remain "
         "unresolved after 4 hours."),
        ("2. Uptime commitment and service credits",
         "Meridian Cloud commits to a monthly uptime of 99.9% for Pro clusters. If monthly uptime falls below 99.9% "
         "but stays at or above 99.0%, the customer receives a service credit of 10% of the monthly fee. If uptime "
         "falls below 99.0%, the credit is 25%. Credits are applied to the next invoice and never paid out in cash.\n"
         "Customers must request a credit within 30 days after the end of the affected month. Scheduled maintenance "
         "is excluded from the uptime calculation. Maintenance windows take place on Sundays between 02:00 and 04:00 "
         "UTC and are announced at least 7 days in advance. Outages caused by customer misconfiguration are also "
         "excluded."),
    ]),
}


def render(path: Path, title: str, pages: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4, pageCompression=0)
    c.setTitle(title)
    w, h = A4
    for heading, body in pages:
        y = h - 60
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"{title}")
        y -= 28
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, heading)
        y -= 22
        c.setFont("Helvetica", 10.5)
        for para in body.split("\n"):
            for line in textwrap.wrap(para, 92):
                c.drawString(50, y, line)
                y -= 15
            y -= 8
        c.showPage()
    c.save()


if __name__ == "__main__":
    for rel, (title, pages) in DOCS.items():
        render(OUT / rel, title, pages)
        print("wrote", OUT / rel)
