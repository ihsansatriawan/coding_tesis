create table point_receivedfrom_from_nohijup_2015
select * from point_histories 
where 
  category = 'Received From' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01' and
  detail not like 'FREE%'

drop table point_transferto_from_2015;
create table point_transferto_from_2015 as
select 
  * 
from 
  point_histories 
where 
  category = 'Transfer To' and 
  owner_type != 'Admin' and
  date(created_at+INTERVAL'7 hour') >= '2015-01-01'

drop table tmp_data_point;
create table tmp_data_point as
select
  cast(id_pengirim as int),
  cast(id_penerima as int),
  count(*) as freq
from
(
  select 
    a.id as id_pengirim, b.owner_id as id_penerima, b.point
  from 
    users a
  inner join
    point_receivedfrom_from_nohijup_2015 b
      on a.email = b.from
) as foo
group by id_pengirim, id_penerima

INSERT INTO tmp_data_point (id_pengirim, id_penerima, freq)
select
  cast(id_pengirim as int),
  cast(id_penerima as int),
  count(*) as freq
from
(
  select 
    b.owner_id as id_pengirim, a.id as id_penerima, ABS(b.point)
  from 
    users a
  inner join
    point_transferto_from_2015 b
      on a.email = b.to
) as foo
group by id_pengirim, id_penerima

drop table tesis_point_user;
create table tesis_point_user as
select
  id_pengirim,
  id_penerima,
  cast((sum(freq)) as bigint) as freq
from
(
  select id_pengirim, id_penerima, freq from tmp_data_point
) as foo
group by id_pengirim, id_penerima