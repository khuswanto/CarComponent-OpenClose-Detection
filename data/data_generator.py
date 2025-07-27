import os
import asyncio

from pathlib import Path
from itertools import combinations
from utils.car3d import Car3D, Door

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


class DataGenerator:

    @staticmethod
    async def horizontal_scan(car3d, page, prefix, y_step):
        # rotate horizontal
        x_steps, delay = 50, 50
        start_x, start_y = 0, 400
        end_x, end_y = 710, 400

        await page.mouse.move(start_x, start_y)
        await page.mouse.down()

        for x_step in range(1, x_steps + 1):
            x = start_x + (end_x - start_x) * x_step / x_steps
            y = start_y + (end_y - start_y) * x_step / x_steps
            await page.mouse.move(x, y)
            await asyncio.sleep(delay / 1000)
            filepath = THIS_PATH / car3d.variant / prefix / f"{y_step}-{x_step}.png"
            await car3d.screenshot(filepath=filepath)

        await page.mouse.up()

    @classmethod
    async def run(cls):
        car3d = Car3D()
        async with car3d.show_page() as page:
            await car3d.hide_button()
            while True:
                command = input("$ ")
                match command:
                    case "ss":
                        await car3d.screenshot(filepath=THIS_PATH / "example.png")

                    case "scan":
                        for n in range(0, len(Door.__members__) + 1):
                            for doors in combinations(Door.__members__.values(), n):  # type: tuple[Door, ...]
                                try:
                                    await car3d.click_doors(page, doors)
                                except Exception as e:
                                    print(e)
                                    input(f'Website Error, please refresh manual and <enter>!')
                                    await car3d.click_doors(page, doors)

                                # start from bottom view
                                await page.mouse.move(0, 400)
                                await page.mouse.down()
                                await page.mouse.move(0, 350)
                                await page.mouse.up()

                                y_steps = 20
                                for y_step in range(0, 120, y_steps):
                                    await cls.horizontal_scan(
                                        car3d,
                                        page,
                                        '-'.join(door.name for door in doors) if doors else 'AllClose',
                                        y_step,
                                    )

                                    # slightly up
                                    await page.mouse.move(0, 400)
                                    await page.mouse.down()
                                    await page.mouse.move(0, 400+y_steps)
                                    await page.mouse.up()

                                # refresh the doors state
                                await page.reload(wait_until='networkidle')
                                await car3d.hide_button()
                    case _:
                        break


if __name__ == '__main__':
    asyncio.run(DataGenerator.run())
