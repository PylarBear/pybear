# 1-6-22 BEAR THIS NEEDS WORK STILL ALL IN VBA, HAS BEEN PARSED CORRECTLY, HASHES FIXED
# NEED TO REFACTOR TO PYTHON



Public Function validate_user_str(string1 As String, string2 As String, user_len As Integer) As String

    Do While True:
        user_input = UCase(InputBox(string1))
        If _
            InStr(1, UCase(string2), user_input) > 0 And _
            Len(user_input) > 0 And _
            Len(user_input) <= user_len _
        Then
            Exit Do
        End If
    Loop

    validate_user_str = user_input

    Exit Function
End Function



Public Function validate_user_codes(string3 As String, string4 As String) As String
    Do While True:
        user_codestring = UCase(InputBox(string3))

        For str_idx = 1 To Len(user_codestring)
            code_char = Mid(user_codestring, str_idx, 1)
            #CHECK USER ENTERED CODES AGAINST ACTIVE CODES
            If InStr(1, UCase(string4), code_char) = 0 Then
                MsgBox ("USER ENTERED BAD STATUS CODE")
                Exit For
            End If
            #CHECK FOR DUPLICATE ENTRY IN USER CODES
            If Len(user_codestring) - Len(Replace(user_codestring, code_char, "YYY")) > 1 Then
                MsgBox ("DUPLICATE CODE ENTERED")
                Exit For
            End If
        Next str_idx

        If str_idx = Len(user_codestring) + 1 Then Exit Do
    Loop

    validate_user_codes = user_codestring

    Exit Function

End Function


Public Function string_sorter(string5 As String, string6 As String) As String
    string_sorter = ""
    For char_idx = 1 To Len(string6)
        char = Mid(string6, char_idx, 1)    # JUST SAVING SPACE
        If InStr(1, string5, char) > 0 Then string_sorter = string_sorter & char
    Next char_idx

    Exit Function
End Function









Sub WORD_COUNTER()    #CTRL-SHIFT-C


#TO UPDATE THE DATA, MUST PASTE MOST RECENT 'SHEET2' AND 'NAICS' OVER EXISTING


Dim AA As Worksheet
Dim BB As Worksheet
Dim CC As Worksheet
Dim DD As Worksheet
Dim EE As Worksheet
Dim FF As Worksheet

Set AA = Sheets("Sheet2")
Set BB = Sheets("NAICS")
Set CC = Sheets("CODES")
Set DD = Sheets("Sheet3")
Set EE = Sheets("NAICS STRUCTURE")
Set FF = Sheets("SUBST_ARR")


Dim naics_str As String, active_codes As String, raw_ripped_code_str As String, rrcs As String, alpha_str As String, _
    new_ripped_code_str As String, srcs As String, default_miss_str As String, default_hit_str As String, _
    char As String, user_override As String, holder As String, new_str As String, other_str As String, _
    hit_str As String, miss_str, emp_hit_str As String, _
    naics_name As String, naics6 As String, naics10 As String, naics_chars As String, _
    hit_word_array_row_counter As String, miss_word_array_row_counter As String, _
    title_str As String, pw As String, clean_title As String, title_char As String, _
    cuco As String

Dim row As Integer, row2 As Integer, psn As Integer, char_idx As Integer, _
    display_row As Integer, ratio_cutoff As Integer, user_naics As Integer, unals As Integer, unale As Integer, _
    naics_len As Integer, all_row As Integer, apps As Integer, hit_or_miss_ind As Integer, _
    jtarc As Integer, subst_row As Integer, _
    hwarc As Integer, mwarc As Integer, fhwa_row As Integer, fmwa_row As Integer, hit_counter As Integer, _
    wncm_row As Integer, wcr_row As Integer, fwcr_row As Integer, wcrrc As Integer



With WorksheetFunction

Range(DD.Range("T1"), DD.Range("AJ10000")).ClearContents

#'********************************************************************************************************************
#'************CREATE "HIT" AND "MISS" CODE STRINGS********************************************************************
#'************TO BE USED TO PARTITION RESPECTIVE JOB TITLE WORDS INTO ARRAYS******************************************

#RIP ACTIVE CODES FROM "CODES" SHEET
row = 2
active_codes = ""
While CC.Cells(row, 1) <> ""
    active_codes = active_codes & CC.Cells(row, 1)
    row = row + 1
Wend
#END RIP ACTIVE CODES FROM "CODES" SHEET



#RIP CODES FROM SHEET2 AND SORT
raw_ripped_code_str = ""
rrcs = raw_ripped_code_str
row = 2
While AA.Cells(row, 11) <> ""
    psn = InStr(1, rrcs, AA.Cells(row, 11))
    If psn = 0 Then rrcs = rrcs & AA.Cells(row, 11)
    row = row + 1
Wend
#SORT
alpha_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ&"
sorted_ripped_code_str = string_sorter(rrcs, alpha_str)
srcs = sorted_ripped_code_str
#END RIP CODES FROM SHEET2 AND SORT



#VERIFY CODES PULLED FROM SHEET2 ARE ALLOWED (ARE IN CODE SHEET)
For char_idx = 1 To Len(srcs)
    char = Mid(srcs, char_idx, 1)
    If InStr(1, active_codes, char) = 0 Then
        Beep
        MsgBox ("INCORRECT CODE IN SHEET 2 CODES" & _
                vbCrLf & vbCrLf & "MISSION ABORT")
        End
    End If
Next char_idx
#END VERIFY SHEET2 CODES




#FOR CREATING DEFAULT STRINGS, ASSUME THAT MISS STRING IS ALWAYS "OXYZ" THEN FILL IN HIT STRING WITH THE
#REMAINING CODES FROM "CODES" SHEET RIP
default_miss_str = "OXYZ"
default_hit_str = ""
For char_idx = 1 To Len(active_codes)
    char = Mid(active_codes, char_idx, 1)     'JUST SAVING SPACE
    If InStr(1, default_miss_str, char) = 0 Then default_hit_str = default_hit_str & char
Next char_idx
#END CREATE DEFAULT CODE STRINGS


Beep

#OVERRIDE DEFAULT STRINGS Y/N
Do While True:
    If validate_user_str("DEFAULT HIT STRING = " & default_hit_str & vbCrLf & _
                        "DEFAULT MISS STRING = " & default_miss_str & vbCrLf & vbCrLf & _
                        "ACCEPT STRINGS (A) OR OVERWRITE ONE OF THEM (O)?", "AO", 1) = "O" Then
        user_override = validate_user_str("OVERWRITE HIT (H) OR MISS (M)?" & vbCrLf & _
                        "(THE OTHER WILL FILL AUTOMATICALLY)", "HM", 1)


        If user_override = "H" Then
            holder = "HIT"
        ElseIf user_override = "M" Then
            holder = "MISS"
        End If


        #CREATE THE USER-SPECIFIED HIT/MISS STRING
        new_str = validate_user_codes("ACTIVE CODES ARE: " & _
                                active_codes & vbCrLf & _
                                "CURRENT DEFAULT HIT STRING IS: " & _
                                default_hit_str & vbCrLf & _
                                "CURRENT DEFAULT MISS STRING IS: " & _
                                default_miss_str & vbCrLf & vbCrLf & _
                                "ENTER NEW " & holder & " STRING:", _
                                active_codes)

        new_str = string_sorter(new_str, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        #END CREATE USER STRING


        #CREATE THE COMPLEMENTARY STRING
        other_str = ""
        For char_idx = 1 To Len(active_codes)
            char = Mid(active_codes, char_idx, 1)
            If InStr(1, new_str, char) = 0 Then other_str = other_str & char
        Next char_idx
        #END CREATE THE COMPLEMENTARY STRING


        If user_override = "H" Then
        hit_str = new_str
        miss_str = other_str
        ElseIf user_override = "M" Then
        hit_str = other_str
        miss_str = new_str
        End If


    Else:       #IF USER IS NOT OVERRIDING DEFAULTS
        hit_str = default_hit_str
        miss_str = default_miss_str
    End If


    If validate_user_str("HIT STRING = " & hit_str & vbCrLf & _
                        "MISS STRING = " & miss_str & vbCrLf & vbCrLf & _
                        "ACCEPT STRINGS? (Y/N)", _
                        "YN", 1) = "Y" _
    Then Exit Do

Loop
#END OVERRIDE DEFAULT STRINGS
#************END CREATE "HIT" AND "MISS" CODE STRINGS***************************************************************
#*******************************************************************************************************************
#*******************************************************************************************************************


display_row = InputBox("ENTER NUMBER OF ROWS TO DISPLAY")

ratio_cutoff = InputBox("ENTER RATIO CUTOFF" & vbCrLf & "(MINIMUM TIMES WORD APPEARS)")


#************CREATE NAICS "DICTIONARY"********************************************************************************

Do While True:
    user_naics = InputBox("ENTER THE NUMBER FOR THE NAICS TO BE USED" & vbCrLf & _
                            "1) NAICS1" & vbCrLf & _
                            "2) NAICS2" & vbCrLf & _
                            "3) NAICS3" & vbCrLf & _
                            "4) NAICS4" & vbCrLf & _
                            "5) NAICS5" & vbCrLf & _
                            "6) NAICS6" & vbCrLf & _
                            "7) NAICS6-SUB" & vbCrLf & _
                            "8) ALL")
    If user_naics >= 1 And user_naics <= 8 Then Exit Do
Loop


#STUFF FOR SETTING UP NAICS LOOP    (NEEDED TO ACCOMODATE THE "ALL" LOOP CAPABILITY)
If user_naics <= 7 Then
    unals = user_naics          # unals = USER_NAICS_ALL_LOOP_START
    unale = user_naics          # unale = USER_NAICS_ALL_LOOP_END
ElseIf user_naics = 8 Then
    unals = 1
    unale = 7
End If


#SETTING NAICS NAME FOR USE LATER IN DISPLAY
If user_naics <= 6 Then
    naics_name = user_naics
ElseIf user_naics = 7 Then
    naics_name = "6-SUB"
ElseIf user_naics = 8 Then
    naics_name = "ALL"
End If



ReDim NAICS_ARRAY(2 To 2) As String     #FOR NAICS TEXT
ReDim NAICS_H_OR_M(2 To 2) As Integer    #COLUMN 2 FOR HIT/MISS ASSGNMT
all_row = 2
For user_naics = unals To unale:

    DoEvents

    If user_naics <= 6 Then
        naics_len = user_naics
    ElseIf user_naics = 7 Then
        naics_len = 10
    End If

    row = 2

    #LOOP THRU FIRST NAICS COLUMN TO FIND # OF ROWS
    For NAICS_row_idx = 2 To 1000:
        If BB.Cells(NAICS_row_idx, 1) = "" Then
        NAICS_last_row = NAICS_row_idx - 1
        Exit For
        End If
    Next NAICS_row_idx

    Do While AA.Cells(row, 1) <> "":

        ReDim Preserve NAICS_ARRAY(2 To all_row) As String
        ReDim Preserve NAICS_H_OR_M(2 To all_row) As Integer

        #GET NAICS TEXT
        naics6 = .VLookup(AA.Cells(row, 4), BB.Range("A2:D" & CStr(NAICS_last_row)), 2, False)
        naicssub = .VLookup(AA.Cells(row, 4), BB.Range("A2:D" & CStr(NAICS_last_row)), 3, False)
        naics10 = naics6 & "-" & naicssub
        naics_chars = Left(naics10, naics_len)
        #HAD TO DO 2 DIFFERENT VLOOKUPS CUZ WHEN IT'S NOT -SUB IT'S WANTS TO LOOKUP A NUMBER OR ELSE IT THROWS
        #AN ERROR AND WHEN IT'S - SUB IT WANTS TO LOOKUP TEXT
        If user_naics <= 6 Then
            NAICS_ARRAY(all_row) = UCase(.VLookup(Int(naics_chars), EE.Range("A2:B23324"), 2, False))
        ElseIf user_naics = 7 Then
            NAICS_ARRAY(all_row) = UCase(.VLookup(naics_chars, EE.Range("A2:B23324"), 2, False))
        End If

        #ASSIGN ROW TO HIT OR MISS CATEGORY
        If InStr(1, hit_str, AA.Cells(row, 11)) > 0 Then
            NAICS_H_OR_M(all_row) = 1
        ElseIf InStr(1, miss_str, AA.Cells(row, 11)) > 0 Then
            NAICS_H_OR_M(all_row) = 0
        End If

        row = row + 1
        all_row = all_row + 1

    Loop

Next user_naics

#END CREATE NAICS "DICTIONARY"
#************************************************************************************************************************

#****BUILD A DICTIONARY OF KEY PHRASES TO SUBST B4 CLEANING, PARSING, AND FINAL KEYWORD SUBSTS***************************

ReDim KP_SUBST_ARR(1 To 100, 1 To 2) As String

keyphraserow = 2
keyphrasecol = 2
keyphrasesubcol = 3
kpsubstarrrow = 1
kpsubstarrcol = 1
kpsubstarrsubcol = 2

Do While True:

    KP_SUBST_ARR(kpsubstarrrow, kpsubstarrcol) = FF.Cells(keyphraserow, keyphrasecol)
    KP_SUBST_ARR(kpsubstarrrow, kpsubstarrsubcol) = FF.Cells(keyphraserow, keyphrasesubcol)

    kpsubstarrrow = kpsubstarrrow + 1
    keyphraserow = keyphraserow + 1

    If FF.Cells(keyphraserow, keyphrasecol) = "" Or _
        kpsubstarrrow > UBound(KP_SUBST_ARR, 1) Then Exit Do

Loop
#****END KEY PHRASE DICTIONARY*******************************************************************************************


#****BUILD A DICTIONARY OF PARSED WORDS TO SUBSTITUTE / IGNORE***********************************************************
ReDim KW_SUBST_ARR(1 To 100, 1 To 2) As String

keywordrow = 2
keywordcol = 10
keywordsubcol = 11
kwsubstarrrow = 1
kwsubstarrcol = 1
kwsubstarrsubcol = 2

Do While True:

    KW_SUBST_ARR(kwsubstarrrow, kwsubstarrcol) = FF.Cells(keywordrow, keywordcol)
    KW_SUBST_ARR(kwsubstarrrow, kwsubstarrsubcol) = FF.Cells(keywordrow, keywordsubcol)

    kwsubstarrrow = kwsubstarrrow + 1
    keywordrow = keywordrow + 1

    If FF.Cells(keywordrow, keywordcol) = "" Then Exit Do

Loop
#****END KW SUBST DICTIONARY***************************************************************************************


#****************************************************************************************************************
#****************************************************************************************************************
#WRANGLE WORDS IN JOB TITLES AND CREATE ARRAYS TO HOLD PARSED WORDS**********************************************
#title = 7
#code = 11

alpha_num_str = alpha_str & "0123456789&_"
For title_or_naics = 1 To 2

    #CREATE JOB TITLE PARSE ARRAY, HIT AND MISS ARRAYS, THEN FINAL SORTED HIT AND MISS ARRAYS
    ReDim JTWA(1 To 1) As String
    ReDim HWA(1 To 2000, 1 To 2) As Variant     #HWA = HIT_WORD_ARRAY
    ReDim MWA(1 To 2000, 1 To 2) As Variant     #MWA = MISS_WORD_ARRAY
    ReDim FHWA(1 To 2000, 1 To 2) As Variant     #FHWA = FINAL_HIT_WORD_ARRAY (SORTED)
    ReDim FMWA(1 To 2000, 1 To 2) As Variant     #FMWA = FINAL_MISS_WORD_ARRAY (SORTED)
    ReDim WNCM(1 To 2000, 1 To 2) As Variant     #WNCM = WORDS_NEVER_CONTACTING_ME
    ReDim WCR(1 To 2000, 1 To 2) As Variant     #WCR = WORD_CONTACT_RATIO
    ReDim FWCR(1 To 2000, 1 To 2) As Variant     #FWCR = FINAL_WORD_CONTACT_RATIO

    #LOOP THRU ALL ROWS OF JOB TITLES OR NAICS WORDS,
    #CLEANING & PARSING WORDS FOR EACH AND DISTRIBUTING TO HIT & MISS ARRAYS

    row = 2

    Do While True:

        DoEvents

        If title_or_naics = 1 Then title_str = UCase(AA.Cells(row, 7))
        If title_or_naics = 2 Then title_str = NAICS_ARRAY(row)

        #FLAG AA.CELLS(ROW,X) OR NAICS_ARRAY(ROW) AS A HIT OR A MISS
        If title_or_naics = 1 Then
            If InStr(1, hit_str, AA.Cells(row, 11)) > 0 Then
                hit_or_miss_ind = 1
            ElseIf InStr(1, miss_str, AA.Cells(row, 11)) > 0 Then hit_or_miss_ind = 0
            Else:
                MsgBox("SHEET2 ROW " & row & " NOT IN hit_str OR miss_str!" & vbCrLf & "MISSION ABORT")
                End
            End If
        ElseIf title_or_naics = 2 Then
            If NAICS_H_OR_M(row) = 1 Then
                hit_or_miss_ind = 1
            ElseIf NAICS_H_OR_M(row) = 0 Then
                hit_or_miss_ind = 0
            Else:
                MsgBox("NAICS_H_OR_M ROW " & row & " IS NOT CORRECTLY ASSGND A hit_or_miss_indicator!" & vbCrLf & "MISSION ABORT")
            End If
        End If

        #LOOK FOR KEY PHRASES IN RAW TITLE STRING & REPLACE
        row_idx = 1
        Do While KP_SUBST_ARR(row_idx, 1) <> "":

            If InStr(1, title_str, KP_SUBST_ARR(row_idx, kpsubstarrcol)) > 0 Then
                title_str = Replace(title_str, KP_SUBST_ARR(row_idx, kpsubstarrcol), KP_SUBST_ARR(row_idx, kpsubstarrsubcol))
            End If

            row_idx = row_idx + 1
        Loop

        #CLEAN JUNK CHAR OUT OF TITLE STRING & TRIM
        For char_idx = 1 To Len(title_str):
            If InStr(1, alpha_num_str, Mid(title_str, char_idx, 1)) = 0 Then
                title_str =.Replace(title_str, char_idx, 1, " ")
            End If
        Next char_idx

        title_str = WorksheetFunction.Trim(title_str)

        #PARSE WORDS IN CLEAN TITLE STR THEN PUT INTO A HOLDING ARRAY B4 ASSGN TO HIT OR MISS ARRAYS
        pw = ""     # pw = parsed_word
        jtarc = 0    # job_title_array_row_counter
        ReDim JTWA(1 To 1) As String
        If Len(title_str) > 0 Then
            For char_idx = 1 To Len(title_str):
                title_char = Mid(title_str, char_idx, 1)
                If title_char <> " " Then pw = pw & title_char
                #LOOK IF WORD (pw) IS IN KW_SUBST_ARR AND SUBST IF DOING JOB TITLES AND NOT NAICS
                If Len(pw) > 0 And _
                    (Mid(title_str, char_idx, 1) = " " Or char_idx = Len(title_str)) Then
                    For subst_row = 1 To UBound(KW_SUBST_ARR)
                        If pw = KW_SUBST_ARR(subst_row, 1) Then pw = KW_SUBST_ARR(subst_row, 2)
                        If KW_SUBST_ARR(subst_row, 1) = "" Then Exit For
                    Next subst_row
                End If

                If Len(pw) > 0 And _
                    (Mid(title_str, char_idx, 1) = " " Or char_idx = Len(title_str)) Then
                    jtarc = jtarc + 1
                    #JTWA = JOB_TITLE_WORD_ARRAY
                    ReDim Preserve JTWA(1 To jtarc) As String
                    JTWA(UBound(JTWA)) = pw
                    pw = ""
                End If
            Next char_idx
            jtarc = 0

            DoEvents

            #PUT WORDS FROM HOLDING ARRAY INTO HIT OR MISS ARRAY BASED ON hit_or_miss_ind FLAG

            #hwarc = hit_word_array_row_counter
            #mwarc = miss_word_array_row_counter
            If hit_or_miss_ind = 1 Then
                For jtarc = LBound(JTWA) To UBound(JTWA):
                    For hwarc = LBound(HWA, 1) To UBound(HWA, 1):
                        If JTWA(jtarc) = HWA(hwarc, 1) Then
                            HWA(hwarc, 2) = HWA(hwarc, 2) + 1
                            Exit For
                        ElseIf HWA(hwarc, 1) = "" Then
                            HWA(hwarc, 1) = JTWA(jtarc)
                            HWA(hwarc, 2) = 1
                            Exit For
                        End If
                    Next hwarc
                Next jtarc
            End If

            DoEvents

            If hit_or_miss_ind = 0 Then
                For jtarc = LBound(JTWA) To UBound(JTWA):
                    For mwarc = LBound(MWA, 1) To UBound(MWA, 1):
                        If JTWA(jtarc) = MWA(mwarc, 1)
                            Then MWA(mwarc, 2) = MWA(mwarc, 2) + 1
                            Exit For
                        ElseIf MWA(mwarc, 1) = "" Then
                            MWA(mwarc, 1) = JTWA(jtarc)
                            MWA(mwarc, 2) = 1
                            Exit For
                        End If
                    Next mwarc
                Next jtarc
            End If

        End If

        row = row + 1

        If title_or_naics = 1 And AA.Cells(row, 1) = "" Then Exit Do
        If title_or_naics = 2 And row > UBound(NAICS_ARRAY) Then Exit Do
    Loop

    DoEvents

    #SORT HIT AND MISS ARRAYS BY WORD COUNT, DESC
    #CREATE 2 NEW ARRAYS FINAL_HIT_ARRAY (FHWA) AND FINAL_MISS_ARRAY (FMWA)

    #STARTING W A HIGH #, LOOK THRU HWA FOR A MATCH THEN PUT INTO FHWA, THEN REPEAT, COUNTING DOWN TO 1
    fhwa_row = 0
    For hit_counter = row To 1 Step - 1     #ASSUME THE MOST TIMES A WORD COULD SHOW UP = # ROWS
        For hwarc = LBound(HWA, 1) To UBound(HWA, 1):
            If HWA(hwarc, 2) = hit_counter Then
                fhwa_row = fhwa_row + 1
                FHWA(fhwa_row, 1) = HWA(hwarc, 1)
                FHWA(fhwa_row, 2) = HWA(hwarc, 2)
            End If
        Next hwarc
    Next hit_counter

    DoEvents

    #STARTING W A HIGH #, LOOK THRU MWA FOR A MATCHING # THEN PUT INTO FMWA, THEN REPEAT, COUNTING DOWN TO 1
    fmwa_row = 0
    For hit_counter = row To 1 Step - 1    #ASSUME THE MOST TIMES A WORD COULD SHOW UP = # ROWS
        For mwarc = LBound(MWA, 1) To UBound(MWA, 1):
            If MWA(mwarc, 2) = hit_counter Then
                fmwa_row = fmwa_row + 1
                FMWA(fmwa_row, 1) = MWA(mwarc, 1)
                FMWA(fmwa_row, 2) = MWA(mwarc, 2)
            End If
        Next mwarc
    Next hit_counter

    #END WRANGLE JOB TITLES, CLASSIFY WORDS, AND SORT************************************************************************

    #CREATE ARRAY OF WORDS HAVING NEVER CONTACTED ME (WNCM)*********************************************************************
    #ALSO CREATE ARRAY THAT CALCULATES WORD CONTACT RATIO (CWCR)... WORD # CONTACTS / (WORD # CONCACTS + WORD # NON-CONTACTS)
    wncm_row = 0
    wcr_row = 0
    For fmwa_row = 1 To UBound(FMWA)

        If FMWA(fmwa_row, 1) = "" Then Exit For

        For fhwa_row = 1 To UBound(FHWA)
            If FHWA(fhwa_row, 1) = FMWA(fmwa_row, 1) _
                And FHWA(fhwa_row, 2) + FMWA(fmwa_row, 2) >= ratio_cutoff Then
                wcr_row = wcr_row + 1
                WCR(wcr_row, 1) = FMWA(fmwa_row, 1)
                # MULTIPLY RATIO BY 10000 FOR FUTURE SORTING PURPOSES
                WCR(wcr_row, 2) = Round(10000 * FHWA(fhwa_row, 2) / (FHWA(fhwa_row, 2) + FMWA(fmwa_row, 2)), 0)
                Exit For
            ElseIf FHWA(fhwa_row, 1) = FMWA(fmwa_row, 1) _
                And FHWA(fhwa_row, 2) + FMWA(fmwa_row, 2) < ratio_cutoff Then
                Exit For
            End If

            If FHWA(fhwa_row, 1) = "" Then
                wncm_row = wncm_row + 1
                WNCM(wncm_row, 1) = FMWA(fmwa_row, 1)
                WNCM(wncm_row, 2) = FMWA(fmwa_row, 2)
                Exit For
            End If

        Next fhwa_row

        If FMWA(fmwa_row, 1) = "" Then Exit For
    Next fmwa_row

    # STARTING W A HIGH #, LOOK THRU WCR FOR A MATCHING # THEN PUT INTO FWCR, THEN REPEAT, COUNTING DOWN TO 1
    fwcr_row = 0
    For hit_counter = 10000 To 1 Step - 1 'BECAUSE RATIO WAS MULTIPLIED BY 10000
        For wcrrc = LBound(WCR, 1) To UBound(WCR, 1):
            If WCR(wcrrc, 2) = hit_counter Then
                fwcr_row = fwcr_row + 1
                FWCR(fwcr_row, 1) = WCR(wcrrc, 1)
                FWCR(fwcr_row, 2) = WCR(wcrrc, 2) / 10000
            End If
        Next wcrrc
    Next hit_counter


    DoEvents


    If title_or_naics = 1 Then wc1 = 20   #wc = write_column
    If title_or_naics = 1 Then wc2 = 21   #wc = write_column
    If title_or_naics = 1 Then wc3 = 22   #wc = write_column

    If title_or_naics = 1 Then wc4 = 25   #wc = write_column
    If title_or_naics = 1 Then wc5 = 26   #wc = write_column
    If title_or_naics = 1 Then wc6 = 27   #wc = write_column



    If title_or_naics = 2 Then wc1 = 29   #wc = write_column
    If title_or_naics = 2 Then wc2 = 30   #wc = write_column
    If title_or_naics = 2 Then wc3 = 31   #wc = write_column

    If title_or_naics = 2 Then wc4 = 34   #wc = write_column
    If title_or_naics = 2 Then wc5 = 35   #wc = write_column
    If title_or_naics = 2 Then wc6 = 36   #wc = write_column


    If title_or_naics = 1 Then word = "JOB TITLE"
    If title_or_naics = 2 Then word = "NAICS" & naics_name

    DD.Cells(1, wc1) = "MOST COMMON " & word & " WORDS CONTACTING ME"
    For display = 1 To.Min(display_row, UBound(FHWA) - LBound(FHWA) + 1):
        DD.Cells(1 + display, wc1) = display
        DD.Cells(1 + display, wc2) = FHWA(display, 1)
        DD.Cells(1 + display, wc3) = FHWA(display, 2)
    Next display

    DD.Cells(1 + display_row + 2, wc1) = "MOST COMMON " & word & " WORDS NOT CONTACTING ME"
    For display = 1 To.Min(display_row, UBound(FMWA) - LBound(FMWA) + 1):
        DD.Cells(1 + display_row + 2 + display, wc1) = display
        DD.Cells(1 + display_row + 2 + display, wc2) = FMWA(display, 1)
        DD.Cells(1 + display_row + 2 + display, wc3) = FMWA(display, 2)
    Next display

    DD.Cells(1 + display_row + 2 + display_row + 2, wc1) = "MOST COMMON " & word & " WORDS NEVER CONTACTING ME"
    For display = 1 To.Min(display_row, UBound(WNCM) - LBound(WNCM) + 1):
        DD.Cells(1 + display_row + 2 + display_row + 2 + display, wc1) = display
        DD.Cells(1 + display_row + 2 + display_row + 2 + display, wc2) = WNCM(display, 1)
        DD.Cells(1 + display_row + 2 + display_row + 2 + display, wc3) = WNCM(display, 2)
    Next display

    DD.Cells(1, wc4) = "HIGHEST " & word & " CONTACT RATIO"
    For display = 1 To.Min(display_row, UBound(FWCR) - LBound(FWCR) + 1):
    DD.Cells(1 + display, wc4) = display
    DD.Cells(1 + display, wc5) = FWCR(display, 1)
    DD.Cells(1 + display, wc6) = FWCR(display, 2)
    Next display


    DD.Cells(1 + display_row + 2, wc4) = "LOWEST " & word & " CONTACT RATIO"


    end_finder = 1
    Do While FWCR(end_finder + 1, 1) <> ""
        end_finder = end_finder + 1
    Loop


    For display = 1 To display_row:   # , UBound(FWCR) - LBound(FWCR) + 1):
        DD.Cells(1 + display_row + 2 + display, wc4) = display
        If display >= UBound(FWCR, 1) Or display > end_finder Then Exit For
        DD.Cells(1 + display_row + 2 + display, wc5) = FWCR(end_finder + 1 - display, 1)
        DD.Cells(1 + display_row + 2 + display, wc6) = FWCR(end_finder + 1 - display, 2)
    Next display




Next title_or_naics

End With

End Sub









