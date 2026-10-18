# Security Incident Response Procedure

This procedure defines how Meridian Cloud detects, classifies and resolves security incidents. It applies to every employee and contractor. Operational outages that are not security related are handled by the on-call process in the engineering handbook.

## Severity levels

Incidents are classified from SEV1 (most severe) to SEV4. A SEV1 security incident is declared when customer data exposure is suspected, or when an attacker has administrative access to a production system. A SEV2 incident involves a confirmed vulnerability that is being actively exploited without evidence of data exposure. SEV3 covers contained events such as a single compromised employee account. SEV4 covers policy violations with no technical impact.

## Roles and escalation

Anyone who suspects an incident must report it in the #sec-incident channel or by calling the security hotline. For a SEV1 incident, the Security Incident Commander must be assigned within 30 minutes of declaration. The Incident Commander owns decisions, appoints a communications lead and a technical lead, and opens a dedicated war room. The Chief Information Security Officer is informed within 1 hour of any SEV1 declaration.

## Containment and evidence

The first priority is containment: isolate affected systems, revoke credentials and rotate secrets. Evidence must be preserved before systems are rebuilt. Disk snapshots and relevant logs are copied to the forensic storage bucket, and the chain of custody is recorded in the incident ticket. Systems must not be powered off unless the Incident Commander approves.

## Customer and regulator notification

When a breach of personal data is confirmed, affected customers and the relevant regulator must be notified within 72 hours of confirmation. The legal team approves all external wording. Notification letters describe what happened, which data was involved, and what customers should do.

## Post-incident review

A blameless post-incident review must be held within 5 business days of resolving a SEV1 or SEV2 incident. The review produces a written report with a timeline, root cause, and action items with named owners and due dates. Action items are tracked until they are closed, and a summary is shared with all engineering staff.
