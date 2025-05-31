#!/usr/bin/env python3
"""Check database saves - verify questions and sessions were saved"""

import asyncio
import os
import asyncpg
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

async def check_recent_saves():
    """Check for recently saved data"""
    print("🔍 Checking database for recent saves...\n")

    connection_string = os.getenv("NEON_DATABASE_URL")
    if not connection_string:
        print("❌ NEON_DATABASE_URL not found")
        return

    conn = await asyncpg.connect(connection_string)

    try:
        # Check recent sessions (last 24 hours)
        print("📊 Recent Generation Sessions:")
        try:
            sessions = await conn.fetch("""
                SELECT session_id::text, config_id, status, questions_generated,
                       questions_approved, created_at
                FROM deriv_generation_sessions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 10
            """)

            if sessions:
                for session in sessions:
                    created = session['created_at'].strftime("%H:%M:%S")
                    session_id_short = str(session['session_id'])[:8]
                    print(f"   🎯 {session_id_short}... | {session['config_id']} | "
                          f"{session['status']} | {session['questions_generated']} questions | {created}")
            else:
                print("   ❌ No recent sessions found")
        except Exception as e:
            print(f"   ❌ Sessions table not accessible: {e}")

        # Check recent questions
        print(f"\n📝 Recent Questions:")
        questions = await conn.fetch("""
            SELECT question_id::text, target_grade, marks, command_word,
                   insertion_status, validation_passed, created_at,
                   LEFT(question_data::text, 100) as preview
            FROM deriv_candidate_questions
            WHERE created_at > NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 10
        """)

        if questions:
            for q in questions:
                created = q['created_at'].strftime("%H:%M:%S")
                status = "✅" if q['validation_passed'] else "❌"
                question_id_short = str(q['question_id'])[:8]
                print(f"   {status} {question_id_short}... | Grade {q['target_grade']} | "
                      f"{q['marks']} marks | {q['command_word']} | {created}")
                print(f"      Preview: {q['preview']}...")
        else:
            print("   ❌ No recent questions found")

        # Check database stats
        print(f"\n📈 Database Stats:")
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_questions,
                COUNT(CASE WHEN validation_passed = true THEN 1 END) as validated_questions,
                COUNT(CASE WHEN insertion_status = 'pending' THEN 1 END) as pending_questions,
                COUNT(CASE WHEN insertion_status = 'auto_approved' THEN 1 END) as approved_questions,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as questions_today,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as questions_this_week
            FROM deriv_candidate_questions
        """)

        print(f"   Total Questions: {stats['total_questions']}")
        print(f"   Validated Questions: {stats['validated_questions']}")
        print(f"   Pending Review: {stats['pending_questions']}")
        print(f"   Approved Questions: {stats['approved_questions']}")
        print(f"   Questions Today: {stats['questions_today']}")
        print(f"   Questions This Week: {stats['questions_this_week']}")

        # Check latest question content
        latest = await conn.fetchrow("""
            SELECT question_data, created_at, target_grade, marks, command_word
            FROM deriv_candidate_questions
            ORDER BY created_at DESC
            LIMIT 1
        """)

        if latest:
            print(f"\n🔍 Latest Question (saved at {latest['created_at'].strftime('%H:%M:%S')}):")
            import json
            data = json.loads(latest['question_data'])
            print(f"   ID: {data.get('question_id_global', 'Unknown')}")
            print(f"   Content: {data.get('raw_text_content', 'No content')}")
            print(f"   Grade: {latest['target_grade']} | Marks: {latest['marks']}")
            print(f"   Command: {latest['command_word']}")
            print(f"   Subject Refs: {data.get('taxonomy', {}).get('subject_content_references', 'Unknown')}")

    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_recent_saves())
