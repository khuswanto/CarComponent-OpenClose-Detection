import os
from io import BytesIO
from enum import Enum
from typing import AsyncIterator
from pathlib import Path
from contextlib import asynccontextmanager

from PIL import Image
from playwright.async_api import async_playwright, Page
from playwright._impl._errors import TimeoutError


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
    def __init__(self, variant: str = 'dark'):
        self.variant = variant

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
            self.page = await browser.new_page()
            await self.page.goto("https://euphonious-concha-ab5c5d.netlify.app/", wait_until="networkidle")
            await self.page.emulate_media(color_scheme='dark')

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

    async def screenshot(self, filepath: Path = None) -> Image:
        new_size = 620
        im_bytes = await self.page.locator('canvas').screenshot()
        im = Image.open(BytesIO(im_bytes))

        # Crop the center of the image
        width, height = im.size
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        im = im.crop((left, top, right, bottom))

        if filepath:
            print(filepath)
            os.makedirs(filepath.parent, exist_ok=True)
            im.save(filepath)

        return im
