from publish.initialize import initialize

initialize()

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import Core
from controller.controller import router

app = FastAPI(**Core.server.fastapiConfig.__dict__)
app.include_router(router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

uvicorn.run(app, host="124.221.73.192", port=8081)
