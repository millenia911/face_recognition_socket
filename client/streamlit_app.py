import streamlit as st
import socketio
import requests
import jsonpickle
from streamlit import session_state as sess

sio = socketio.Client()
html_page = open("./stream.html", "r")

def init_session_states():
    state_list = ["role", "socket_connected"]
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
        # img_file_buffer = st.camera_input("Take a picture")
        sub = st.form_submit_button("Submit")
        if sub:
            if len(files) > 0 and len(name) > 3:
                pbar = st.progress(0)
                num = (int(100/len(files)))
                for n, f in enumerate(files):
                    data_bytes = f.getvalue()
                    _, status = submit_image(name, data_bytes)
                    if status != 200:
                        st.warning(f"Something is wrong when uploading {f.name}")
                        continue
                    pbar.progress((n+1)*num)
                st.success(F"Pictures for {name} is submited!")
            else: st.warning("Image should be attached and name must be longer than 3 character")
            
    _back = st.button("Back")
    if _back:
        sess["role"] = None
        st.experimental_rerun()

def streamer_page():
    st.components.v1.html(html_page.read(), height=500)
    _back = st.button("Back")
    if _back:
        sess["role"] = None
        st.experimental_rerun()

def submit_image(name, img_bytes:bytes):
    print("sending picture for", name)
    req = {"name":name, 
            "image_bytes":img_bytes}
    res = requests.post("http://127.0.0.1:8888/api/admin", 
                        json=jsonpickle.encode(req))
    return res.text, res.status_code

def connect_socketio():
    print("connecting to socket server...")
    print("please wait")
    sio.connect("http://127.0.0.1:8888", 
                    wait_timeout=200, 
                    namespaces=["/admin", "/streamer"])
    print("SID is ", sio.sid)


if __name__ == "__main__":
    init_session_states()
    if sess["socket_connected"]:
        st.text("connected to socketio server")
    if sess["role"] is None:
        login_page()

    elif sess["role"] == "Admin":
        admin_page()
    
    elif sess["role"] == "Streamer":
        streamer_page()

    else: raise ValueError("Invalid session role")
