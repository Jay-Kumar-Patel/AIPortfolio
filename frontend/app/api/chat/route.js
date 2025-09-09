import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const { question } = await request.json();

    if (!question || question.trim() === '') {
      return NextResponse.json(
        { error: 'Question is required' },
        { status: 400 }
      );
    }

    // Make the actual API call to your AI service
    const apiResponse = await fetch('http://localhost:3001/api/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Add any required API keys or authentication headers
      },
      body: JSON.stringify({ question }),
    });

    if (!apiResponse.ok) {
      throw new Error(`API responded with status: ${apiResponse.status}`);
    }

    const data = await apiResponse.json();
    
    // Extract and clean the answer from the API response
    // This assumes your API returns data in the format shown in your example
    const rawAnswer = data.answer || '';
    
    return NextResponse.json({
      response: rawAnswer
    });

  } catch (error) {
    console.error('Error in chat API:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}