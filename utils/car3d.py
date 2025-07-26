from enum import Enum
from typing import AsyncIterator
from pathlib import Path
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright, Page


class Door(Enum):
    FrontLeft = 0
    FrontRight = 1
    RearLeft = 2
    RearRight = 3
    Hood = 4


class Car3D:
    """
    Run js to get (x, y):
    document.onmousemove = (e) => {
      console.log(`clientX: ${e.clientX}, clientY: ${e.clientY}`); // relative to viewport
      console.log(`pageX: ${e.pageX}, pageY: ${e.pageY}`); // relative to whole page (scroll)
    };
    """
    @property
    def page(self) -> Page:
        return self._page

    @page.setter
    def page(self, page: Page):
        self._page = page

    @asynccontextmanager
    async def show_page(self, **kwargs) -> AsyncIterator:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            self.page = await browser.new_page(viewport={"width": 800, "height": 800}, **kwargs)
            await self.page.goto("https://euphonious-concha-ab5c5d.netlify.app/", wait_until="networkidle")

            yield self.page

            await self.page.close()
            await browser.close()

    async def hide_button(self):
        await self.page.evaluate("""
            const element = document.querySelector('#root div div:nth-of-type(2)');
            if (element) {
                element.style.display = 'none';
            }
        """)

    async def show_button(self):
        await self.page.evaluate("""
            const element = document.querySelector('#root div div:nth-of-type(2)');
            if (element) {
                element.style.display = '';
            }
        """)

    async def click_doors(self, page, doors: tuple[Door, ...]):
        await self.show_button()
        for door in doors:
            await page.locator("button").nth(door.value).click()
        await self.hide_button()

    async def screenshot(self, filepath: Path = None) -> bytes:
        if filepath:
            print(filepath)

        x_offset = 70
        width = 725 - x_offset
        return await self.page.screenshot(
            path=filepath,
            clip={"x": x_offset, "y": 45, "width": width, "height": width}
        )
