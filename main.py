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
        event_type = post_data.get("event") or post_data.get("event_type")
        
        print(f"📥 Received Webhook Event: {event_type}", flush=True)
        
        if event_type == "call_analyzed":
            
            # 1. FIX THE NESTED JSON TRAP
            # Safely drill down to the actual call object regardless of how Retell packages it
            call_obj = {}
            if "data" in post_data and "call" in post_data["data"]:
                call_obj = post_data["data"]["call"]
            elif "call" in post_data:
                call_obj = post_data["call"]
            else:
                call_obj = post_data
            
            call_id = call_obj.get("call_id", "Unknown")
            status = call_obj.get("status", "Unknown")
            
            # 2. FILTER OUT NO-ANSWER & DEAD CALLS
            # Instantly drop calls where the customer didn't pick up
            if status in ["no-answer", "failed", "canceled", "busy", "machine"] or call_obj.get("duration", 0) == 0:
                print(f"⏭️ Skipped QA: Call {call_id} was unanswered (Status: {status}).", flush=True)
                return JSONResponse(status_code=200, content={"message": "Skipped unconnected call"})

            # 3. GET AGENT IDENTIFIER
            agent_name = call_obj.get("agent_name")
            agent_id = call_obj.get("agent_id")
            
            # Use name if available, otherwise use ID
            identifier = agent_name if agent_name else agent_id

            if not identifier:
                print(f"⚠️ Warning: No agent_name or agent_id found for connected call {call_id}.", flush=True)
                return JSONResponse(status_code=200, content={"message": "Skipped - No identifier"})

            agent_config = get_agent_config(identifier)
            
            if not agent_config:
                print(f"⏭️ Skipped QA: No matching config found for agent/ID '{identifier}'.", flush=True)
                return JSONResponse(status_code=200, content={"message": "Skipped"})

            # 4. EXTRACT TRANSCRIPT
            transcript_obj = call_obj.get("transcript_object", [])
            if not transcript_obj and "call_analysis" in call_obj:
                transcript_obj = call_obj["call_analysis"].get("transcript_object", [])

            if not transcript_obj:
                print(f"⚠️ Warning: Transcript is empty for connected call {call_id}! Skipping.", flush=True)
                return JSONResponse(status_code=200, content={"message": "Skipped empty transcript"})

            transcript_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in transcript_obj])

            # Trigger Background QA Task
            background_tasks.add_task(process_qa_task, call_id, transcript_text, agent_config, identifier)

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
            .pass { border-left-color: #10b981; } .fail { border-left-color: #ef4444; } .partial { border-left-color: #f59e0b; }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Past Calls</h2>
            <button onclick="clearHistory()" style="width:100%; margin-bottom:15px; padding:8px; cursor:pointer; border:1px solid #ddd; background:#fff; border-radius:4px;">🗑️ Clear Local History</button>
            <div id="call-list">Loading...</div>
        </div>
        <div id="content">
            <h2 style="color: #888;">Select a call from the left to view its QA Scorecard.</h2>
            <div id="scorecard-view"></div>
        </div>

        <script>
            let allReports = [];

            // 1. Load existing reports from localStorage when the page opens
            function loadLocalHistory() {
                const saved = localStorage.getItem("qa_reports_history");
                if (saved) {
                    allReports = JSON.parse(saved);
                }
            }

            // 2. Fetch fresh reports from the server and merge them
            function fetchAndMergeReports() {
                fetch('/api/reports')
                    .then(res => res.json())
                    .then(serverData => {
                        // Merge server data with local data, avoiding duplicates based on call_id
                        let mergedMap = new Map();
                        
                        // Add local reports first
                        allReports.forEach(report => mergedMap.set(report.call_id, report));
                        
                        // Overwrite/Add server reports
                        serverData.forEach(report => mergedMap.set(report.call_id, report));

                        // Convert Map back to array and sort by date descending
                        allReports = Array.from(mergedMap.values());
                        allReports.sort((a, b) => new Date(b.date) - new Date(a.date));

                        // Save the newly merged list back to localStorage
                        localStorage.setItem("qa_reports_history", JSON.stringify(allReports));

                        renderSidebar();
                    })
                    .catch(err => {
                        console.error("Could not fetch server reports, falling back to local history.", err);
                        renderSidebar();
                    });
            }

            // 3. Clear button functionality
            function clearHistory() {
                if (confirm("Are you sure you want to clear your local QA history?")) {
                    localStorage.removeItem("qa_reports_history");
                    allReports = [];
                    renderSidebar();
                    document.getElementById('scorecard-view').innerHTML = "";
                }
            }

            // Draw the list on the left
            function renderSidebar() {
                const listDiv = document.getElementById('call-list');
                
                if (allReports.length === 0) {
                    listDiv.innerHTML = "<p style='color:#666;'>No QA reports generated yet.</p>";
                    return;
                }

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

            // Draw the full scorecard on the right
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

            // Initialize
            loadLocalHistory();
            fetchAndMergeReports();
            
            // Auto-refresh the server fetch every 30 seconds to look for new calls
            setInterval(fetchAndMergeReports, 30000); 
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
