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

# Create a folder to store our reports
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

async def process_qa_task(call_id: str, transcript_text: str, agent_config: dict, agent_name: str):
    """Runs the LLM and saves the report with a timestamp."""
    print(f"🔍 Starting QA evaluation for call {call_id}...")
    
    scorecard_json = await qa_engine.evaluate_call(transcript_text, agent_config)
    
    if scorecard_json:
        # Parse the JSON so we can inject the date and agent name
        data = json.loads(scorecard_json)
        data["call_id"] = call_id
        data["agent_name"] = agent_name
        data["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to the reports folder
        filename = f"{REPORTS_DIR}/{call_id}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ QA Complete! Saved to {filename}")


@app.post("/qa-webhook")
async def handle_qa_webhook(request: Request, background_tasks: BackgroundTasks):
    """Retell sends the call data here when a call ends."""
    try:
        post_data = await request.json()
        event_type = post_data.get("event")
        
        if event_type == "call_analyzed":
            call_data = post_data.get("data", {})
            call_id = call_data.get("call_id")
            agent_name = call_data.get("agent_name", "Unknown Agent")

            agent_config = get_agent_config(agent_name)
            if not agent_config:
                return JSONResponse(status_code=200, content={"message": "Skipped - Unknown Agent"})

            transcript_obj = call_data.get("transcript_object", [])
            transcript_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in transcript_obj])

            # Run in background
            background_tasks.add_task(process_qa_task, call_id, transcript_text, agent_config, agent_name)

        return JSONResponse(status_code=200, content={"received": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


# ======================================================================
# DASHBOARD ROUTES (For the Dev Team to view past scorecards)
# ======================================================================

@app.get("/api/reports")
async def get_all_reports():
    """Returns a list of all saved scorecards."""
    reports = []
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(REPORTS_DIR, filename), "r") as f:
                reports.append(json.load(f))
    # Sort newest first
    reports.sort(key=lambda x: x.get("date", ""), reverse=True)
    return reports

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the HTML UI."""
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

            // 1. Fetch all reports from the server
            fetch('/api/reports')
                .then(res => res.json())
                .then(data => {
                    allReports = data;
                    renderSidebar();
                });

            // 2. Draw the list on the left
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

            // 3. Draw the full scorecard on the right when clicked
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