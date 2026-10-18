# Production Deployment Runbook

This runbook describes how engineers at Meridian Cloud ship changes to production. It applies to every service that serves customer traffic.

## Deployment windows

Production deployments are allowed from Tuesday to Thursday. A change freeze applies from Friday 12:00 until Monday 09:00 UTC, and during the winter shutdown. Emergency fixes during a freeze need approval from the on-call engineering manager and must be linked to an incident ticket.

## Approvals

Every production change requires two approvals: one code review approval and one approval from a member of the service owning team who is not the author. Database schema migrations additionally need a review from the database reliability team before the change is merged. Changes that touch authentication or billing require a security review.

## Canary rollout

Deployments use a staged rollout. The new version first receives 5% of traffic for 30 minutes, then 25% for 30 minutes, and finally 100%. During each stage the deployment tool compares error rate and latency against the previous version. The rollout pauses automatically if p99 latency rises by more than 20%.

## Automatic rollback

The deployment tool rolls back automatically when the error rate exceeds 2% over a 5 minute window. Engineers can also roll back manually with the command meridianctl rollback --service NAME, which restores the previous version within about 3 minutes. After any rollback the engineer must open an incident ticket and add the failed version to the release notes.

## After the deployment

The deploying engineer stays online and watches the dashboards for at least 60 minutes after reaching 100% traffic. Release notes are posted in the #releases channel. Feature flags that are no longer needed must be removed within 30 days of full rollout to avoid dead code paths.
