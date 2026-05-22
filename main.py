import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from retell import Retell
from dotenv import load_dotenv

from qa_engine import QAEngine
from agent_configs import get_agent_config

load_dotenv()

app = FastAPI()
retell = Retell(api_key=os.environ["RETELL_API_KEY"])
qa_engine = QAEngine()

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

async def process_qa_task(call_id: str, transcript_text: str, agent_config: dict, agent_name: str):
    print(f"🔍 Starting LLM QA evaluation for call {call_id}...", flush=True)
    
    scorecard_json = await qa_engine.evaluate_call(transcript_text, agent_config)
    
    if scorecard_json:
        try:
            data = json.loads(scorecard_json)
            data["call_id"] = call_id
            data["agent_name"] = agent_name
            data["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            filename = f"{REPORTS_DIR}/{call_id}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            print(f"✅ QA Complete! Saved to {filename}", flush=True)
        except Exception as e:
            print(f"❌ Failed to parse LLM JSON output: {e}", flush=True)
            print(f"Raw Output: {scorecard_json}", flush=True)
    else:
        print(f"❌ LLM failed to return a scorecard for {call_id}", flush=True)


@app.post("/qa-webhook")
async def handle_qa_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        post_data = await request.json()
        event_type = post_data.get("event")
        
        print(f"📥 Received Webhook Event: {event_type}", flush=True)
        
        if event_type == "call_analyzed":
            # 1. Print the raw payload so we can see exactly what Retell is sending
            print(f"📦 RAW PAYLOAD: {json.dumps(post_data)[:500]}...", flush=True)
            
            # 2. Fix: Check BOTH 'call' and 'data' keys for the payload
            call_data = post_data.get("call") or post_data.get("data") or post_data
            
            call_id = call_data.get("call_id")
            agent_name = call_data.get("agent_name")
            agent_id = call_data.get("agent_id")
            
            if not agent_name:
                print(f"⚠️ Warning: No agent_name found! Available keys are: {list(call_data.keys())}", flush=True)
                # If we only have the ID, we will print it to help us debug
                agent_name = "Unknown Agent"

            agent_config = get_agent_config(agent_name)
            
            if not agent_config:
                print(f"⏭️ Skipped QA: No matching config found for agent '{agent_name}'.", flush=True)
                return JSONResponse(status_code=200, content={"message": "Skipped"})

            # Extract transcript
            transcript_obj = call_data.get("transcript_object", [])
            
            # Sometimes Retell nests the transcript inside 'call_analysis'
            if not transcript_obj and "call_analysis" in call_data:
                transcript_obj = call_data["call_analysis"].get("transcript_object", [])

            transcript_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in transcript_obj])

            if not transcript_text.strip():
                print(f"⚠️ Warning: Transcript is empty for {call_id}!", flush=True)

            background_tasks.add_task(process_qa_task, call_id, transcript_text, agent_config, agent_name)

        return JSONResponse(status_code=200, content={"received": True})
    except Exception as e:
        print(f"❌ Webhook Crash: {e}", flush=True)
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/api/reports")
async def get_all_reports():
    reports = []
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(REPORTS_DIR, filename), "r") as f:
                reports.append(json.load(f))
    reports.sort(key=lambda x: x.get("date", ""), reverse=True)
    return reports

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QA Master Dashboard</title>
        <style>
            body { font-family: system-ui; display: flex; margin: 0; height: 100vh; background: #f4f7f6;}
            #sidebar { width: 300px; background: white; border-right: 1px solid #ddd; overflow-y: auto; padding: 20px; }
            #content { flex: 1; padding: 40px; overflow-y: auto; }
            .report-card { padding: 15px; border: 1px solid #eee; margin-bottom: 10px; border-radius: 8px; cursor: pointer; transition: 0.2s;}
            .report-card:hover { background: #f8fafc; border-color: #3b82f6;}
            .score-badge { display: inline-block; padding: 5px 10px; border-radius: 20px; font-weight: bold; color: white; float: right;}
            .bg-good { background: #10b981; } .bg-ok { background: #f59e0b; } .bg-bad { background: #ef4444; }
            .section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #ccc;}
            .pass { border-left-color: #10b981; } .fail { border-left-color: #ef4444; }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Past Calls</h2>
            <div id="call-list">Loading...</div>
        </div>
        <div id="content">
            <h2 style="color: #888;">Select a call from the left to view its QA Scorecard.</h2>
            <div id="scorecard-view"></div>
        </div>

        <script>
            let allReports = [];
            fetch('/api/reports')
                .then(res => res.json())
                .then(data => {
                    allReports = data;
                    renderSidebar();
                });

            function renderSidebar() {
                const listDiv = document.getElementById('call-list');
                listDiv.innerHTML = allReports.map((r, index) => {
                    let color = r.overall_compliance_score > 80 ? 'bg-good' : (r.overall_compliance_score > 60 ? 'bg-ok' : 'bg-bad');
                    return `
                    <div class="report-card" onclick="viewReport(${index})">
                        <span class="score-badge ${color}">${r.overall_compliance_score}</span>
                        <strong>${r.agent_name}</strong><br>
                        <small style="color:#666;">${r.date}</small>
                    </div>
                    `;
                }).join('');
            }

            function viewReport(index) {
                const data = allReports[index];
                const sectionsHtml = data.sections.map(sec => `
                    <div class="section ${sec.status.toLowerCase()}">
                        <h3>${sec.category} <span style="float:right; color:#666;">${sec.status}</span></h3>
                        <p><strong>Reasoning:</strong> ${sec.reasoning}</p>
                        <p style="background: #f8fafc; padding: 10px; font-style: italic;">
                            " ${sec.transcript_moments.join('<br>" ')} "
                        </p>
                    </div>
                `).join('');

                document.getElementById('scorecard-view').innerHTML = `
                    <h1>Scorecard: ${data.agent_name}</h1>
                    <p><strong>Date:</strong> ${data.date} | <strong>Call ID:</strong> ${data.call_id}</p>
                    <p><strong>Lead Classification:</strong> ${data.lead_classification}</p>
                    <p><strong>Captured Fields:</strong> ${JSON.stringify(data.captured_fields)}</p>
                    <hr style="border: 1px solid #eee; margin: 20px 0;">
                    ${sectionsHtml}
                `;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
