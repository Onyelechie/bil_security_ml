from fastapi import APIRouter

router = APIRouter(prefix="/api/heartbeat", tags=["heartbeat"])

@router.post("")
def heartbeat():
    # TODO: Implement edge PC heartbeat
    return {"message": "Heartbeat received (not implemented)"}
