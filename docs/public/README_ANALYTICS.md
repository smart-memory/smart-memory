# Google Analytics 4 Setup for SmartMemory Documentation

This documentation site includes Google Analytics 4 (GA4) integration for user analytics and insights.

## Configuration

### 1. Get Your GA4 Tracking ID

1. Go to [Google Analytics](https://analytics.google.com/)
2. Create a new property or use an existing one
3. Set up a GA4 data stream for your website
4. Copy your **Measurement ID** (format: `G-XXXXXXXXXX`)

### 2. Set Environment Variable

For **local development**:
```bash
# Create .env file in docs/public directory
echo "GA_TRACKING_ID=G-YOUR-ACTUAL-ID" > docs/public/.env
```

For **production deployment**:
```bash
# Set environment variable in your deployment system
export GA_TRACKING_ID=G-YOUR-ACTUAL-ID
```

### 3. Deployment Scripts

The `push_docs` script automatically uses the `GA_TRACKING_ID` environment variable during build.

```bash
# Set your GA4 tracking ID before deployment
export GA_TRACKING_ID=G-YOUR-ACTUAL-ID
./push_docs
```

## Privacy Features

- **IP Anonymization**: Enabled by default (`anonymizeIP: true`)
- **GDPR Compliance**: Consider adding a cookie consent banner if required
- **Data Retention**: Configure in GA4 settings according to your needs

## Analytics Data Available

With GA4 enabled, you'll get insights on:

- **Page Views**: Most visited documentation pages
- **User Behavior**: How users navigate through the docs
- **Search Queries**: What users search for in the docs
- **Geographic Data**: Where your users are located
- **Device Information**: Desktop vs mobile usage
- **Referral Sources**: How users find your documentation

## Verifying Setup

1. Deploy with your GA4 tracking ID
2. Visit your documentation site
3. Check the browser's Network tab for `gtag` requests
4. Verify data appears in GA4 Real-time reports (may take a few minutes)

## Example Environment Setup

```bash
# For SmartMemory docs deployment
export GA_TRACKING_ID=G-ABC123DEF4
export S3_BUCKET=your-docs-bucket
./push_docs
```

## Security Notes

- Never commit your actual tracking ID to version control
- Use environment variables for production deployments
- The placeholder `G-XXXXXXXXXX` will be used if no environment variable is set
