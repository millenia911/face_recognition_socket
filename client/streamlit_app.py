import streamlit as st
from streamlit import session_state as sess

def init_session_states():
    state_list = ["role"]
    for _state in state_list:
        if _state not in sess:
            sess[_state] = None

def login_page():
    with st.form("login"):
        rl = st.radio("Select role", ["Admin", "Streamer"])
        sub = st.form_submit_button("Login")
        if sub:
            sess["role"] = rl
            st.experimental_rerun()

def admin_page():
    st.header("Register new person")
    with st.form("submit_form"):
        name = st.text_input("Input name")
        files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        img_file_buffer = st.camera_input("Take a picture")
        sub = st.form_submit_button("Submit")
        if sub:
            if len(files) > 0 and len(name) > 3:
                pbar = st.progress(0)
                num = (int(100/len(files)))
                for n, f in enumerate(files):
                    data_bytes = f.read()
                    submit_image(name, data_bytes)
                    pbar.progress((n+1)*num)
                st.info(F"Pictures for {name} is submited!")
            else: st.warning("Image should be attached and name must be longer than 3 character")
    pass

def streamer_page():
    pass

def submit_image(name, data):
    print(name)



if __name__ == "__main__":
    init_session_states()
    if sess["role"] is None:
        login_page()

    elif sess["role"] == "Admin":
        admin_page()
    
    elif sess["role"] == "Streamer":
        streamer_page()

    else: raise ValueError("Invalid session role")
