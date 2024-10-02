"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import dill

from rxconfig import config
from typing import List
from reflex_project.predictions import TextSuggestion

text_suggestion = TextSuggestion()

class KeyBoard(rx.State):
    current_text: str = ''
    current_predictions: List[str] = text_suggestion.suggest_text(''.split())[0]

    def make_prediction(self):
        self.current_predictions = text_suggestion.suggest_text(self.current_text.split())[0]

    def write_letter(self, letter: str):
        self.current_text += letter
        self.make_prediction()

    def delete_letter(self):
        self.current_text = self.current_text[:-1]
        self.make_prediction()

    def handle_input_change(self, value: str):
        self.current_text = value
        self.make_prediction()

    def accept_prediction(self, value: str):
        if len(self.current_text) > 0 and self.current_text[-1].isalpha():
            last_non_letter_index = len(self.current_text) - 1
            while last_non_letter_index >= 0 and self.current_text[last_non_letter_index].isalpha():
                last_non_letter_index -= 1
            self.current_text =  self.current_text[:last_non_letter_index + 1]
        self.current_text += value

        self.make_prediction()


def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.container(
        rx.flex(
            rx.foreach(
                KeyBoard.current_predictions,
                lambda prediction: rx.button(
                    rx.text(prediction),
                    on_click=KeyBoard.accept_prediction(prediction),
                )
            ),
            direction="row",  
        ),
        rx.heading(KeyBoard.current_text),
        rx.input(
            placeholder="KeyBoard possibilities...",
            value=KeyBoard.current_text, 
            on_change=KeyBoard.handle_input_change
        ),
        rx.flex(
            rx.flex(
                rx.button("1", on_click=KeyBoard.write_letter("1")),
                rx.button("2", on_click=KeyBoard.write_letter("2")),
                rx.button("3", on_click=KeyBoard.write_letter("3")),
                rx.button("4", on_click=KeyBoard.write_letter("4")),
                rx.button("5", on_click=KeyBoard.write_letter("5")),
                rx.button("6", on_click=KeyBoard.write_letter("6")),
                rx.button("7", on_click=KeyBoard.write_letter("7")),
                rx.button("8", on_click=KeyBoard.write_letter("8")),
                rx.button("9", on_click=KeyBoard.write_letter("9")),
                rx.button("0", on_click=KeyBoard.write_letter("0")),
                rx.button("-", on_click=KeyBoard.write_letter("-")),
                rx.button("=", on_click=KeyBoard.write_letter("+")),
                rx.button(rx.icon("step_back"), on_click=KeyBoard.delete_letter()),
                flex_wrap="nowrap",
                direction="row",
            ),
            rx.flex(
                rx.button("Q", on_click=KeyBoard.write_letter("Q")),
                rx.button("W", on_click=KeyBoard.write_letter("W")),
                rx.button("E", on_click=KeyBoard.write_letter("E")),
                rx.button("R", on_click=KeyBoard.write_letter("R")),
                rx.button("T", on_click=KeyBoard.write_letter("T")),
                rx.button("Y", on_click=KeyBoard.write_letter("Y")),
                rx.button("U", on_click=KeyBoard.write_letter("U")),
                rx.button("I", on_click=KeyBoard.write_letter("I")),
                rx.button("O", on_click=KeyBoard.write_letter("O")),
                rx.button("P", on_click=KeyBoard.write_letter("P")),
                flex_wrap="nowrap",
                direction="row",
            ),
            rx.flex(
                rx.button("A", on_click=KeyBoard.write_letter("A")),
                rx.button("S", on_click=KeyBoard.write_letter("S")),
                rx.button("D", on_click=KeyBoard.write_letter("D")),
                rx.button("F", on_click=KeyBoard.write_letter("F")),
                rx.button("G", on_click=KeyBoard.write_letter("G")),
                rx.button("H", on_click=KeyBoard.write_letter("H")),
                rx.button("J", on_click=KeyBoard.write_letter("J")),
                rx.button("K", on_click=KeyBoard.write_letter("K")),
                rx.button("L", on_click=KeyBoard.write_letter("L")),
            ),
            rx.flex(
                rx.button("`", on_click=KeyBoard.write_letter("`")),
                rx.button("Z", on_click=KeyBoard.write_letter("Z")),
                rx.button("X", on_click=KeyBoard.write_letter("X")),
                rx.button("C", on_click=KeyBoard.write_letter("C")),
                rx.button("V", on_click=KeyBoard.write_letter("V")),
                rx.button("B", on_click=KeyBoard.write_letter("B")),
                rx.button("N", on_click=KeyBoard.write_letter("N")),
                rx.button("M", on_click=KeyBoard.write_letter("M")),
                rx.button(",", on_click=KeyBoard.write_letter(",")),
                rx.button(".", on_click=KeyBoard.write_letter(".")),
                rx.button("?", on_click=KeyBoard.write_letter("?")),
                align="center",
            ),
            rx.flex(
                rx.button("space", on_click=KeyBoard.write_letter(" ")),
                align="center",
            ),
            direction="column",
            align="center",
        ),  
    )

app = rx.App()
app.add_page(index)
