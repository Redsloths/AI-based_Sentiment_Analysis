📌# AI-based_Sentiment_Analysis
영화를 보는 관객의 표정 변화와 몰입도를 AI로 분석하여 점수로 변환.

1. 제작 기간 & 참여 인원
2026년 4월 14일 ~ 5월 13일
팀 프로젝트

2. 사용 기술
python
yolo8n
PyTorch
Pandas
NumPy

3. 핵심 기능
이 서비스의 핵심 기능은 컨텐츠 등록 기능입니다.
사용자는 단지 컨텐츠의 카테고리를 선택하고, URL만 입력하면 끝입니다.
이 단순한 기능의 흐름을 보면, 서비스가 어떻게 동작하는지 알 수 있습니다.

핵심 기능 설명 펼치기

5. 핵심 트러블 슈팅
5.1. 컨텐츠 필터와 페이징 처리 문제
저는 이 서비스가 페이스북이나 인스타그램 처럼 가볍게, 자주 사용되길 바라는 마음으로 개발했습니다.
때문에 페이징 처리도 무한 스크롤을 적용했습니다.

하지만 무한스크롤, 페이징 혹은 “더보기” 버튼? 어떤 걸 써야할까 라는 글을 읽고 무한 스크롤의 단점들을 알게 되었고,
다양한 기준(카테고리, 사용자, 등록일, 인기도)의 게시물 필터 기능을 넣어서 이를 보완하고자 했습니다.

그런데 게시물이 필터링 된 상태에서 무한 스크롤이 동작하면,
필터링 된 게시물들만 DB에 요청해야 하기 때문에 아래의 기존 코드 처럼 각 필터별로 다른 Query를 날려야 했습니다.

기존 코드
이 때 카테고리(tag)로 게시물을 필터링 하는 경우,
각 게시물은 최대 3개까지의 카테고리(tag)를 가질 수 있어 해당 카테고리를 포함하는 모든 게시물을 질의해야 했기 때문에
아래 개선된 코드와 같이 QueryDSL을 사용하여 다소 복잡한 Query를 작성하면서도 페이징 처리를 할 수 있었습니다.
개선된 코드

6. 그 외 트러블 슈팅
npm run dev 실행 오류
vue-devtools 크롬익스텐션 인식 오류 문제
ElementUI input 박스에서 `v-on:keyup.enter="메소드명"`이 정상 작동 안하는 문제
Post 목록 출력시에 Member 객체 출력 에러
프로젝트를 git init으로 생성 후 발생하는 npm run dev/build 오류 문제
태그 선택후 등록하기 누를 때 `object references an unsaved transient instance - save the transient instance before flushing` 오류
JSON: Infinite recursion (StackOverflowError)
H2 접속문제
컨텐츠수정 모달창에서 태그 셀렉트박스 드랍다운이 뒤쪽에 보이는 문제
HTTP delete Request시 개발자도구의 XHR(XMLHttpRequest )에서 delete요청이 2번씩 찍히는 이유
이미지 파싱 시 og:image 경로가 달라서 제대로 파싱이 안되는 경우
구글 로그인으로 로그인한 사용자의 정보를 가져오는 방법이 스프링 2.0대 버전에서 달라진 것
랭킹 동점자 처리 문제

6. 회고 / 느낀점
프로젝트 개발 회고 글: https://zuminternet.github.io/ZUM-Pilot-integer/
