from fastapi import APIRouter

from controller import image, user

router = APIRouter()
router.include_router(image.router, prefix="/image", tags=["image"])
router.include_router(user.router, prefix="/user", tags=["user"])
