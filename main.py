import streamlit as st

from ui import standard_derivatives_tab, fractional_derivatives_tab, display_tips


def main():
    st.title("Calculateur de Dérivées")

    tabs = st.tabs(["Dérivées Standards", "Dérivées Fractionnaires"])

    with tabs[0]:
        standard_derivatives_tab()

    with tabs[1]:
        fractional_derivatives_tab()

    display_tips()


if __name__ == "__main__":
    main()
