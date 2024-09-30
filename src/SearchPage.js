import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const SearchPage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [answer, setAnswer] = useState(''); // New state to store the answer

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  const handleSearch = async () => {
    const query = fileContent || searchQuery; // Use file content if available, otherwise use search query

    if (!query) {
      setAnswer('Please enter a search query or upload a file.');
      return;
    }

    try {
      const response = await axios.post('https://hhxlbkmuuk.execute-api.us-east-1.amazonaws.com/main/question', {
        query: query,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('Success:', response.data);

      // Extract and store the answer in the state
      setAnswer(response.data.Answer);
    } catch (error) {
      console.error('Error:', error);
      setAnswer('An error occurred while searching.');
    }
  };

  const handleClear = () => {
    setSearchQuery('');
    setFileContent('');
    setAnswer('');
    // Clear the file input value
    document.getElementById('file-input').value = '';
  };

  return (
    <div className="SearchPage">
      <h1>Search Page</h1>
      <div className="input-group">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter search query"
          className="input-field"
        />
        <input
          type="file"
          accept=".txt"
          onChange={handleFileChange}
          className="input-file"
          id="file-input"
        />
      </div>
      <div className="button-group">
        <button
          onClick={handleSearch}
          className="search-button"
        >
          Search
        </button>
        <button
          onClick={handleClear}
          className="clear-button"
        >
          Clear
        </button>
      </div>
      {answer && (
        <div className="Answer" style={{ marginTop: '20px' }}>
          <h2>Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
};

export default SearchPage;