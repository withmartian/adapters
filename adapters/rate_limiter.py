from abc import ABC, abstractmethod

from aiolimiter import AsyncLimiter


class AbstractRateLimiter(ABC):
    @abstractmethod
    async def acquire(self, *args, **kwargs):
        pass

    @abstractmethod
    async def release(self):
        pass


class OpenAIModelRateLimiter(AbstractRateLimiter):
    def __init__(self, rpm: int = 0):
        self.limiter = AsyncLimiter(rpm) if rpm > 0 else None

    async def acquire(self, *args, **kwargs):
        if self.limiter:
            await self.limiter.acquire()

    async def release(self):
        pass
