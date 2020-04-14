import streamlit as st

def main():
    st.title('Hello World!')
    st.header('This is a reader')
    st.subheader('This is a subheader')
    st.text('This is a text')
    st.image('logo.png')

    button = st.button("Hi, I'm a button")
    if button:
        st.markdown('Clicked')

    check = st.checkbox("Hi, I'm a check box")
    if check:
        st.markdown('Marked')

    radio = st.radio('Choice a option:', ('Option 1', 'Option 2', 'Option 3'))
    if radio == 'Option 1':
        st.markdown('Option 1 selected')
    if radio == 'Option 2':
        st.markdown('Option 2 selected')
    if radio == 'Option 3':
        st.markdown('Option 3 selected')

    select = st.selectbox('Choice a option:', ('Option 1', 'Option 2'))
    if select == 'Option 1':
        st.markdown('Option 1 selected')
    if select == 'Option 2':
        st.markdown('Option 2 selected')

    multi = st.multiselect('Choice one or more options:', ('Option 1', 'Option 2'))
    if multi is not None:
        st.markdown(multi)

    fileup = st.file_uploader('Upload your file here', type='csv')
    if fileup is not None:
        st.markdown('File uploaded')

if __name__ == '__main__':
    main()
