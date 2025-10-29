# AION NEXUS API Documentation

## Overview

The AION NEXUS platform provides a comprehensive REST API for integration with existing industrial systems. This documentation covers endpoints, authentication, data formats, and integration examples.

## 🚀 Quick Start

### Base URL
```
https://api.aion-nexus.com/v1
```

### Authentication
All API requests require authentication using an API key:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.aion-nexus.com/v1/status
```

## 📡 Core Endpoints

### 1. Prediction API

#### Submit Vibration Data for Analysis
```http
POST /predict
```

**Request Body:**
```json
{
  "bearing_id": "BRG_0234",
  "timestamp": "2025-10-27T14:23:45Z",
  "data": {
    "vibration": [0.012, 0.013, ...],  // Array of vibration samples
    "sampling_rate": 25600,            // Hz
    "duration": 0.16                   // seconds
  },
  "context": {
    "load_percent": 85,
    "temperature_c": 72,
    "rpm": 1800,
    "hours_since_lubrication": 1200
  }
}
```

**Response:**
```json
{
  "prediction": {
    "fault_type": "outer_race",
    "confidence": 0.978,
    "severity": "moderate",
    "remaining_life_days": 30
  },
  "physics_validation": {
    "confirmed": true,
    "harmonics_found": 4,
    "agreement_score": 0.873
  },
  "recommendation": {
    "action": "schedule_maintenance",
    "urgency": "medium",
    "timeframe_days": 30
  }
}
```

### 2. Batch Processing

#### Submit Multiple Bearings
```http
POST /predict/batch
```

**Request Body:**
```json
{
  "bearings": [
    {
      "bearing_id": "BRG_0234",
      "data": {...}
    },
    {
      "bearing_id": "BRG_0235",
      "data": {...}
    }
  ]
}
```

### 3. Historical Analysis

#### Get Historical Predictions
```http
GET /history/{bearing_id}
```

**Query Parameters:**
- `start_date`: ISO 8601 date
- `end_date`: ISO 8601 date
- `limit`: Maximum results (default: 100)

**Response:**
```json
{
  "bearing_id": "BRG_0234",
  "history": [
    {
      "timestamp": "2025-10-27T14:23:45Z",
      "fault_type": "normal",
      "confidence": 0.95
    },
    ...
  ]
}
```

### 4. Health Monitoring

#### System Status
```http
GET /status
```

**Response:**
```json
{
  "status": "operational",
  "version": "6.5.0",
  "uptime_hours": 720,
  "predictions_today": 1247,
  "average_latency_ms": 14.8
}
```

## 🔧 Integration Guides

### SCADA Integration

#### OPC UA Configuration
```python
# Example: OPC UA client configuration
from aion_nexus import OPCUAConnector

connector = OPCUAConnector(
    server_url="opc.tcp://localhost:4840",
    api_key="YOUR_API_KEY"
)

# Subscribe to bearing data
connector.subscribe("ns=2;i=1234", callback=process_bearing_data)
```

#### Modbus TCP Integration
```python
# Example: Modbus TCP integration
from aion_nexus import ModbusConnector

connector = ModbusConnector(
    host="192.168.1.100",
    port=502,
    api_key="YOUR_API_KEY"
)

# Read vibration data from registers
data = connector.read_registers(start=1000, count=100)
```

### MES Integration

#### SAP Integration
```abap
* Example: SAP ABAP integration
DATA: lo_client TYPE REF TO if_http_client,
      lv_json   TYPE string.

cl_http_client=>create_by_url(
  url = 'https://api.aion-nexus.com/v1/predict'
  client = lo_client ).

lo_client->request->set_header_field(
  name = 'Authorization'
  value = 'Bearer YOUR_API_KEY' ).
```

## 📊 Data Formats

### Vibration Data Format

**Supported Formats:**
- Raw time-series (recommended)
- FFT spectrum
- Envelope spectrum

**Data Requirements:**
- Sampling rate: ≥10 kHz (25.6 kHz optimal)
- Duration: ≥0.1 seconds
- Resolution: 16-bit minimum

### Context Data Format

**Required Fields:**
```json
{
  "load_percent": 0-200,        // Percentage of rated load
  "temperature_c": -40 to 150,  // Celsius
  "rpm": 0-10000,               // Rotations per minute
  "hours_since_lubrication": 0-10000
}
```

**Optional Fields:**
```json
{
  "ambient_temp_c": -40 to 60,
  "humidity_percent": 0-100,
  "vibration_overall_mm_s": 0-100,
  "production_code": "string",
  "operator_notes": "string"
}
```

## 🔐 Authentication & Security

### API Key Management

Request an API key:
```bash
curl -X POST https://api.aion-nexus.com/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "company": "Your Company",
       "email": "tech@company.com"
     }'
```

### Rate Limits

| Plan | Requests/Minute | Requests/Day | Batch Size |
|------|----------------|--------------|------------|
| Trial | 10 | 100 | 5 |
| Standard | 60 | 10,000 | 20 |
| Enterprise | Unlimited | Unlimited | 100 |

### Security Best Practices

1. **Never expose API keys in client-side code**
2. **Use HTTPS for all requests**
3. **Rotate keys regularly**
4. **Implement IP whitelisting for production**

## 🔄 Webhooks

### Configuration
```json
{
  "webhook_url": "https://your-system.com/webhook",
  "events": ["fault_detected", "severity_change"],
  "secret": "your_webhook_secret"
}
```

### Event Types

| Event | Trigger | Payload |
|-------|---------|---------|
| `fault_detected` | New fault identified | Prediction results |
| `severity_change` | Severity level changes | Old/new severity |
| `maintenance_due` | RUL < threshold | Bearing details |

## 📈 Performance Metrics

### Latency Expectations

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Single prediction | 15ms | 25ms | 50ms |
| Batch (10 bearings) | 80ms | 150ms | 300ms |
| Historical query | 50ms | 200ms | 500ms |

### Throughput

- **Single instance**: 1,000 predictions/minute
- **Clustered deployment**: 10,000+ predictions/minute
- **Auto-scaling available** for enterprise plans

## 🔨 SDKs & Libraries

### Official SDKs

| Language | Package | Installation |
|----------|---------|--------------|
| Python | `aion-nexus` | `pip install aion-nexus` |
| Node.js | `@aion-nexus/client` | `npm install @aion-nexus/client` |
| Java | `com.aionnexus:client` | Maven/Gradle |
| C# | `AionNexus.Client` | NuGet |

### Python Example
```python
from aion_nexus import Client

client = Client(api_key="YOUR_API_KEY")

# Submit prediction
result = client.predict(
    bearing_id="BRG_0234",
    vibration_data=data,
    context={
        "load_percent": 85,
        "temperature_c": 72
    }
)

print(f"Fault: {result.fault_type}")
print(f"Confidence: {result.confidence}")
```

## 🧪 Testing & Sandbox

### Sandbox Environment
```
https://sandbox.aion-nexus.com/v1
```

**Test API Key**: `test_key_xxx` (rate limited)

### Sample Data
Download sample vibration data for testing:
- [Normal bearing](https://sandbox.aion-nexus.com/samples/normal.csv)
- [Inner race fault](https://sandbox.aion-nexus.com/samples/inner.csv)
- [Outer race fault](https://sandbox.aion-nexus.com/samples/outer.csv)

## 📚 Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_DATA",
    "message": "Sampling rate must be at least 10kHz",
    "details": {
      "provided": 5000,
      "minimum": 10000
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `RATE_LIMITED` | 429 | Too many requests |
| `INVALID_DATA` | 400 | Data validation failed |
| `BEARING_NOT_FOUND` | 404 | Unknown bearing ID |
| `INTERNAL_ERROR` | 500 | Server error |

## 🆘 Support

### Technical Support
- **Email**: api-support@aion-nexus.com
- **Response time**: <24 hours
- **Documentation**: https://docs.aion-nexus.com

### Community
- **GitHub Discussions**: https://github.com/aion-nexus/aion-nexus/discussions
- **Stack Overflow**: Tag `aion-nexus`

---

**API Version**: 1.0
**Last Updated**: October 2025
**Status**: Production Ready

*Note: Full API access requires a commercial license. Contact daniel.culotta@gmail.com for pricing.*