"""MedEvidence AI - Streamlit 登录 / 注册入口。

启动方式::

    streamlit run login.py

登录成功后调用 :func:`webui.main` 进入主问答界面。
"""

from __future__ import annotations

import streamlit as st

# 必须在其他 Streamlit 页面指令之前执行
st.set_page_config(
    page_title="MedEvidence AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

from user_data_storage import (
    Credentials,
    credentials,
    storage_file,
    write_credentials,
)
from webui import main


APP_CSS = """
<style>
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --cyan: #06b6d4;
    --bg: #f4f9fc;
    --card: #ffffff;
    --text: #172033;
    --muted: #7b8798;
    --line: #e4ebf3;
}

html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 78% 12%, rgba(37, 99, 235, .06), transparent 28%),
        linear-gradient(180deg, #f8fcfe 0%, #f3f8fb 100%);
    color: var(--text);
}

/* 收紧主区宽度，形成参考图中的居中登录卡片 */
.main .block-container {
    max-width: 1180px;
    padding-top: 8.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,.96);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.brand-wrap {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 26px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 23px;
    background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
    box-shadow: 0 8px 20px rgba(37,99,235,.22);
}

.brand-name {
    font-size: 18px;
    line-height: 1.2;
    font-weight: 750;
    color: #172033;
}

.brand-sub {
    margin-top: 4px;
    color: #8a96a7;
    font-size: 12px;
}

.side-caption {
    color: #9aa5b5;
    font-size: 12px;
    margin: 4px 0 10px;
}

.login-title {
    font-size: 28px;
    font-weight: 760;
    color: #172033;
    margin-bottom: 8px;
}

.login-subtitle {
    color: #8a96a7;
    font-size: 14px;
    margin-bottom: 18px;
}

.login-badge {
    display: inline-block;
    color: #2563eb;
    background: #eff6ff;
    border: 1px solid #dbeafe;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 12px;
    margin: 0 5px 7px 0;
}

.login-note {
    text-align: center;
    color: #9aa5b5;
    font-size: 12px;
    margin-top: 18px;
}

/* Streamlit form 卡片 */
[data-testid="stForm"] {
    background: rgba(255,255,255,.98);
    border: 1px solid #e6edf5;
    border-radius: 20px;
    padding: 30px 30px 22px;
    box-shadow: 0 18px 45px rgba(31,65,114,.09);
}

div[data-testid="stTextInput"] input {
    border-radius: 10px;
    min-height: 44px;
    border: 1px solid #dce5ef;
    background: #fbfdff;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #60a5fa;
    box-shadow: 0 0 0 1px #60a5fa;
}

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 44px;
    border-radius: 10px;
    border: 0;
    color: white;
    font-weight: 700;
    background: linear-gradient(90deg, #2f6fed 0%, #2563eb 100%);
    box-shadow: 0 7px 16px rgba(37,99,235,.22);
}

div[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: 0;
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #f7f9fc;
    padding: 8px 10px;
    border-radius: 9px;
    margin-bottom: 4px;
}
</style>
"""


def _init_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "admin" not in st.session_state:
        st.session_state.admin = False
    if "usname" not in st.session_state:
        st.session_state.usname = ""


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-wrap">
                <div class="brand-icon">⚕</div>
                <div>
                    <div class="brand-name">MedEvidence AI</div>
                    <div class="brand-sub">医疗知识智能助手</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-caption">工作区</div>', unsafe_allow_html=True)
        app_mode = st.radio(
            "账户操作",
            ["登录", "注册"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("Knowledge Graph · Vector RAG")
        st.caption("Evidence Grounding · Hybrid Retrieval")
        st.markdown(
            """
            <div style="position:relative;margin-top:18px;color:#9aa5b5;font-size:12px;line-height:1.7;">
            本系统用于医疗知识检索与技术演示，回答内容不替代专业医生诊断。
            </div>
            """,
            unsafe_allow_html=True,
        )

    return app_mode


def login_page() -> None:
    left, center, right = st.columns([1.35, 1.1, 1.35])
    with center:
        with st.form("login_form"):
            st.markdown(
                """
                <div class="login-title">登录 MedEvidence AI</div>
                <div class="login-subtitle">
                    登录后即可进入多源医疗知识问答工作台。
                </div>
                <div style="margin-bottom:8px;">
                    <span class="login-badge">Knowledge Graph</span>
                    <span class="login-badge">Vector RAG</span>
                    <span class="login-badge">Evidence</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            username = st.text_input(
                "用户名",
                value="",
                placeholder="请输入用户名",
            )
            password = st.text_input(
                "密码",
                value="",
                type="password",
                placeholder="请输入密码",
            )
            submit = st.form_submit_button("登录")

            if submit:
                user_cred = credentials.get(username)
                if user_cred and user_cred.password == password:
                    st.session_state.logged_in = True
                    st.session_state.admin = user_cred.is_admin
                    st.session_state.usname = username
                    st.rerun()
                else:
                    st.error("用户名或密码错误，请重新输入。")

        st.markdown(
            '<div class="login-note">登录后可使用会话、知识检索与证据追溯功能</div>',
            unsafe_allow_html=True,
        )


def register_page() -> None:
    left, center, right = st.columns([1.35, 1.1, 1.35])
    with center:
        with st.form("register_form"):
            st.markdown(
                """
                <div class="login-title">创建账户</div>
                <div class="login-subtitle">
                    注册普通用户账户后即可体验医疗知识问答。
                </div>
                """,
                unsafe_allow_html=True,
            )
            new_username = st.text_input(
                "设置用户名",
                value="",
                placeholder="请输入用户名",
            )
            new_password = st.text_input(
                "设置密码",
                value="",
                type="password",
                placeholder="请输入密码",
            )
            register_submit = st.form_submit_button("注册")

            if register_submit:
                if not new_username.strip() or not new_password:
                    st.error("用户名和密码不能为空。")
                elif new_username in credentials:
                    st.error("用户名已存在，请使用其他用户名。")
                else:
                    # SECURITY: 当前仍为 demo 明文存储；生产部署应替换为加盐哈希
                    new_user = Credentials(new_username, new_password, False)
                    credentials[new_username] = new_user
                    write_credentials(storage_file, credentials)
                    st.success(f"用户 {new_username} 注册成功，请在左侧切换到“登录”。")


_init_state()
st.markdown(APP_CSS, unsafe_allow_html=True)

if __name__ == "__main__":
    if not st.session_state.logged_in:
        mode = _render_sidebar()
        if mode == "登录":
            login_page()
        else:
            register_page()
    else:
        main(st.session_state.admin, st.session_state.usname)
