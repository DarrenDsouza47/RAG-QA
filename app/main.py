import streamlit as st
from Utils import (initialize_llm, process_pdf, create_embeddings,
                     create_vectorstore)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def main():
    st.title("PDF Q&A System with Document Retrieval")

    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_pdf is not None:
        st.write("Processing the PDF...")
        documents, warning = process_pdf(uploaded_pdf)

        if warning:
            st.warning(warning)
        elif documents is not None and len(documents) > 0:
            embeddings = create_embeddings(documents)
            vectorstore = create_vectorstore(documents, embeddings)

            llm = initialize_llm()

            ### langchain retrival code
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}")
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain=create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 1}), combine_docs_chain)
            ###


            user_question = st.text_input("Ask a question about the document:")

            if user_question:
                st.write("Getting the answer...")
                result = retrieval_chain.invoke({"input": user_question})

                st.subheader("Answer:")
                st.write(result["answer"])

                st.subheader("Retrieved Documents:")
                for doc in result["context"]:
                    st.write(doc.page_content)
        else:
            st.error("Error: No valid text was extracted from the PDF.")

if __name__ == "__main__":
    main()
